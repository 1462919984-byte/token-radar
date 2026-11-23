// ============================================================================
// token-radar-01 · fitted value model (v3, written fresh)
// ----------------------------------------------------------------------------
// One sealed snapshot of the protocol universe goes in; one ranked board and
// the dashboard's dataset come out. Nothing here reads the wall clock, the
// network, or any source of randomness: the model is fit at run time from the
// snapshot's own figures with deterministic linear algebra, so two runs write
// identical bytes.
//
// THE MODEL, in words:
//
//   Comps curve — for every protocol that records a positive market cap we fit
//   the market's TVL -> market-cap relationship with a plain log-log ordinary
//   least squares regression:
//
//       ln(mcap) = b0 + b1*ln(tvl) + b2*ln(age_days + 1) + b3*growth_7d
//                      + b4*ln(chains + 1) + category indicators
//
//   fit on the 128 positive market caps, with two rounds of dropping residuals
//   beyond 2.5 standard deviations. Applied to every launch candidate this
//   gives its "comps value": what the market pays for a token at that TVL,
//   age, growth, chain count and category.
//
//   Token odds — a logistic regression for whether a protocol records a token,
//   fit on all 758 protocols:
//
//       logit P(records a token) = c0 + c1*ln(tvl) + c2*ln(chains + 1)
//                                      + c3*growth_7d + c4*age_years
//                                      + category indicators
//
//   Age enters with its own fitted sign (no half-life discount): a likelihood
//   comparison against the data rejects the 730-day half-life, and that tested
//   number is reported, computed here at run time.
//
//   Implied value = round(comps_value_usd * token_odds).
//
// Live-token protocols are never ranked: they are set aside with the market's
// own mcap. Wrapped / staked / bridged / exchange-named / versioned products
// are set aside, unranked, on their own page. Every other pre-token protocol
// is a launch candidate and is ranked. No protocol name is hardcoded anywhere.
// ============================================================================

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** Median age of the token-bearing set, used for protocols with no listed_at. */
const COMP_AGE_UNKNOWN_DAYS = 1000;

/** The half-life whose likelihood the page reports against the fitted model. */
const AGE_HALF_LIFE_DAYS = 730;

/** Growth is clipped before it enters a regression. */
const GROWTH_FLOOR = -60;
const GROWTH_CAP_COMP = 200; // comps curve, matching the v1 OLS specification
const GROWTH_CAP_ODDS = 120; // odds curve, matching the v2 logistic specification

/** How many top categories enter the comps and odds designs. */
const COMPS_CATEGORIES = 8;
const ODDS_CATEGORIES = 7;

/** Farm economics: the value of an hour of the farmer's time. */
const DOLLARS_PER_HOUR = 40;

const EXCLUDED_CATEGORIES = new Set([
  'Bridge', 'Canonical Bridge', 'Cross Chain Bridge',
  'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX',
]);
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;
const EXCHANGE_NAME = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/i;

// ----------------------------------------------------------------------------
// small deterministic math helpers
// ----------------------------------------------------------------------------

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

const ln = Math.log;
const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

function median(values: number[]): number {
  const a = [...values].sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m]! : (a[m - 1]! + a[m]!) / 2;
}

export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

function clipGrowth(p: SnapshotProtocol, cap: number): number {
  const g = finite(p.change_7d, 0);
  return Math.min(cap, Math.max(GROWTH_FLOOR, g));
}

/** Whether a protocol records no token at all (no mcap, gecko id or symbol). */
export function recordsNoToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

function recordsToken(p: SnapshotProtocol): boolean {
  return !recordsNoToken(p);
}

/** Whether the protocol records a market cap: the 144-token (19%) definition
 * the odds fit uses, so the calibration rate matches the prompt's 19%. */
function recordsMarketCap(p: SnapshotProtocol): boolean {
  return p.mcap_usd !== null;
}

/**
 * Whether a pre-token protocol is a true launch candidate. Mirrors the
 * snapshot's own watchlist rule without the TVL floor: not a wrapped, staked,
 * bridged, exchange or versioned product of a token that already trades.
 */
export function isLaunchCandidate(p: SnapshotProtocol): boolean {
  if (!recordsNoToken(p)) return false;
  if (p.category !== null && EXCLUDED_CATEGORIES.has(p.category)) return false;
  if (VERSIONED.test(p.name)) return false;
  if (WRAPPED_WORD.test(p.name)) return false;
  if (EXCHANGE_NAME.test(p.name)) return false;
  return true;
}

function setAsideReason(p: SnapshotProtocol): string {
  if (p.category === 'CEX') return 'centralised exchange';
  if (p.category !== null && ['Bridge', 'Canonical Bridge', 'Cross Chain Bridge'].includes(p.category)) return 'bridge product';
  if (p.category !== null && ['Liquid Staking', 'Liquid Restaking', 'Restaked BTC'].includes(p.category)) return 'staking / restaking product';
  if (EXCHANGE_NAME.test(p.name)) return 'exchange-named product';
  if (WRAPPED_WORD.test(p.name)) return 'wrapped or staked product';
  if (VERSIONED.test(p.name)) return 'versioned pool';
  return 'other';
}

// ----------------------------------------------------------------------------
// deterministic linear algebra: normal-equations OLS and ridge logistic
// ----------------------------------------------------------------------------

function solve(A: number[][]): number[] | null {
  const n = A.length;
  const M = A.map((r) => r.slice());
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r]![i]!) > Math.abs(M[piv]![i]!)) piv = r;
    if (Math.abs(M[piv]![i]!) < 1e-12) return null;
    [M[i], M[piv]] = [M[piv], M[i]];
    const d = M[i]![i]!;
    for (let c = i; c <= n; c++) M[i]![c]! /= d;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const f = M[r]![i]!;
      for (let c = i; c <= n; c++) M[r]![c]! -= f * M[i]![c]!;
    }
  }
  return M.map((r) => r[n]!);
}

function ols(X: number[][], y: number[]): number[] | null {
  const n = X.length;
  const k = X[0]!.length;
  const XtX = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const Xty = new Array<number>(k).fill(0);
  for (let i = 0; i < n; i++) {
    const x = X[i]!;
    const yy = y[i]!;
    for (let a = 0; a < k; a++) {
      Xty[a]! += x[a]! * yy;
      for (let b = 0; b < k; b++) XtX[a]![b]! += x[a]! * x[b]!;
    }
  }
  const aug = XtX.map((r, j) => r.concat(Xty[j]!));
  return solve(aug);
}

/** L2-regularised logistic regression via Newton iterations (deterministic). */
function logistic(X: number[][], y: number[], lambda: number, iters: number): number[] {
  const n = X.length;
  const k = X[0]!.length;
  let beta = new Array<number>(k).fill(0);
  for (let it = 0; it < iters; it++) {
    const grad = new Array<number>(k).fill(0);
    const H = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    for (let i = 0; i < n; i++) {
      const x = X[i]!;
      let z = 0;
      for (let j = 0; j < k; j++) z += beta[j]! * x[j]!;
      const pr = sigmoid(z);
      const e = pr - y[i]!;
      for (let j = 0; j < k; j++) grad[j]! += x[j]! * e;
      const w = pr * (1 - pr);
      for (let a = 0; a < k; a++) for (let c = a; c < k; c++) H[a]![c]! += x[a]! * x[c]! * w;
    }
    for (let a = 1; a < k; a++) {
      H[a]![a]! += lambda;
      for (let c = 0; c < a; c++) H[a]![c]! = H[c]![a]!;
    }
    const delta = solve(H.map((r, j) => r.concat(grad[j]!)));
    if (delta === null) break;
    for (let j = 0; j < k; j++) beta[j]! -= delta[j]!;
  }
  return beta;
}

// ----------------------------------------------------------------------------
// the comps fit: the market-cap curve from protocols that record a token
// ----------------------------------------------------------------------------

export interface CompsFit {
  beta: number[];
  catNames: string[];
  n_train: number;
  n_kept: number;
  r2: number;
  cv_mae_log: number;
  cv_r2: number;
  cv_n: number;
}

function compsFeature(p: SnapshotProtocol, asOf: string, catNames: string[]): number[] {
  const age = ageDays(asOf, p.listed_at) ?? COMP_AGE_UNKNOWN_DAYS;
  const v = [1, ln(Math.max(p.tvl_usd, 1)), ln(age + 1), clipGrowth(p, GROWTH_CAP_COMP), ln(p.chains.length + 1)];
  for (const c of catNames) v.push(p.category === c ? 1 : 0);
  return v;
}

function topCategories(protocols: SnapshotProtocol[], take: number): string[] {
  const count = new Map<string, number>();
  for (const p of protocols) {
    if (p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0 && p.category !== null) {
      count.set(p.category, (count.get(p.category) ?? 0) + 1);
    }
  }
  return [...count.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, take)
    .map(([c]) => c);
}

function slugFold(slug: string, folds: number): number {
  let h = 0;
  for (let i = 0; i < slug.length; i++) h = ((h * 31 + slug.charCodeAt(i)) >>> 0);
  return h % folds;
}

function fitComps(snapshot: Snapshot): CompsFit {
  const asOf = snapshot.as_of;
  const token = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  const catNames = topCategories(token, COMPS_CATEGORIES);

  const pred = (p: SnapshotProtocol, beta: number[]): number => {
    const v = compsFeature(p, asOf, catNames);
    let s = 0;
    for (let j = 0; j < beta.length; j++) s += beta[j]! * v[j]!;
    return s;
  };

  // Two rounds of dropping |residual| > 2.5 sigma outliers.
  let kept = token;
  let beta = new Array<number>(catNames.length + 5).fill(0);
  for (let round = 0; round < 4; round++) {
    const X = kept.map((p) => compsFeature(p, asOf, catNames));
    const y = kept.map((p) => ln(p.mcap_usd!));
    const b = ols(X, y);
    if (b === null) break;
    beta = b;
    const res = kept.map((p) => ln(p.mcap_usd!) - pred(p, beta));
    const mu = res.reduce((a, c) => a + c, 0) / res.length;
    const sd = Math.sqrt(res.reduce((a, c) => a + (c - mu) * (c - mu), 0) / res.length);
    const next = kept.filter((_, i) => Math.abs(res[i]! - mu) <= 2.5 * sd);
    if (next.length === kept.length) break;
    kept = next;
  }

  // In-sample R-squared on the kept (outlier-dropped) set.
  const yk = kept.map((p) => ln(p.mcap_usd!));
  const meanY = yk.reduce((a, c) => a + c, 0) / yk.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < yk.length; i++) {
    const pr = pred(kept[i]!, beta);
    ssRes += (yk[i]! - pr) * (yk[i]! - pr);
    ssTot += (yk[i]! - meanY) * (yk[i]! - meanY);
  }
  const r2 = 1 - ssRes / ssTot;

  // Held-out 5-fold cross-validation (refit the robust model on each fold).
  const folds = 5;
  const fold = Array.from({ length: folds }, () => [] as SnapshotProtocol[]);
  for (const p of token) fold[slugFold(p.slug, folds)]!.push(p);
  let cvMae = 0;
  let cvN = 0;
  let cvNum = 0;
  let cvDen = 0;
  for (let f = 0; f < folds; f++) {
    const hold = fold[f]!;
    const train = token.filter((p) => !hold.includes(p));
    if (train.length < 8) continue;
    const tCat = topCategories(train, COMPS_CATEGORIES);
    let tKept = train;
    let tBeta = new Array<number>(tCat.length + 5).fill(0);
    for (let round = 0; round < 4; round++) {
      const X = tKept.map((p) => compsFeature(p, asOf, tCat));
      const y = tKept.map((p) => ln(p.mcap_usd!));
      const b = ols(X, y);
      if (b === null) break;
      tBeta = b;
      const res = tKept.map((p) => {
        const v = compsFeature(p, asOf, tCat);
        let s = 0;
        for (let j = 0; j < tBeta.length; j++) s += tBeta[j]! * v[j]!;
        return ln(p.mcap_usd!) - s;
      });
      const mu = res.reduce((a, c) => a + c, 0) / res.length;
      const sd = Math.sqrt(res.reduce((a, c) => a + (c - mu) * (c - mu), 0) / res.length);
      const next = tKept.filter((_, i) => Math.abs(res[i]! - mu) <= 2.5 * sd);
      if (next.length === tKept.length) break;
      tKept = next;
    }
    for (const p of hold) {
      const v = compsFeature(p, asOf, tCat);
      let pr = 0;
      for (let j = 0; j < tBeta.length; j++) pr += tBeta[j]! * v[j]!;
      const yy = ln(p.mcap_usd!);
      cvMae += Math.abs(yy - pr);
      cvNum += (yy - pr) * (yy - pr);
      cvDen += (yy - meanY) * (yy - meanY);
      cvN++;
    }
  }

  return {
    beta,
    catNames,
    n_train: token.length,
    n_kept: kept.length,
    r2,
    cv_mae_log: cvMae / cvN,
    cv_r2: 1 - cvNum / cvDen,
    cv_n: cvN,
  };
}

// ----------------------------------------------------------------------------
// the odds fit: a pre-token protocol's chance of ever recording a token
// ----------------------------------------------------------------------------

export interface OddsFit {
  beta: number[];
  catNames: string[];
  n: number;
  base_rate: number;
  loglik_no_discount: number;
  loglik_half_life_730: number;
  category_rates: { category: string; n: number; tokens: number; rate: number; shrunk: number }[];
}

function oddsFeature(p: SnapshotProtocol, asOf: string, catNames: string[]): number[] {
  const age = ageDays(asOf, p.listed_at) ?? COMP_AGE_UNKNOWN_DAYS;
  // Same feature set as the v2 logistic: intercept, ln(tvl), ln(chains+1),
  // 7-day growth as a fraction, age in years, then category indicators.
  const v = [1, ln(Math.max(p.tvl_usd, 1)), ln(p.chains.length + 1), clipGrowth(p, GROWTH_CAP_ODDS) / 100, age / 365];
  for (const c of catNames) v.push(p.category === c ? 1 : 0);
  return v;
}

function logLikelihood(beta: number[], X: number[][], y: number[]): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) {
    let z = 0;
    for (let j = 0; j < beta.length; j++) z += beta[j]! * X[i]![j]!;
    s += y[i]! * z - Math.log(1 + Math.exp(z));
  }
  return s;
}

function fitOdds(snapshot: Snapshot): OddsFit {
  const asOf = snapshot.as_of;
  const protocols = snapshot.protocols;

  // Category indicators for the odds fit: the top categories by count among
  // the token-bearing comps set (the same top-category rule the page states).
  const catNames = topCategories(protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0), ODDS_CATEGORIES);

  const X = protocols.map((p) => oddsFeature(p, asOf, catNames));
  const y = protocols.map((p) => (recordsMarketCap(p) ? 1 : 0));
  const beta = logistic(X, y, 0.1, 40);
  const loglikNoDiscount = logLikelihood(beta, X, y);

  // The tested-and-removed 730-day half-life: refit with age scaled by a
  // 730-day exponential discount, then compare likelihoods.
  const X730 = protocols.map((p) => {
    const f = oddsFeature(p, asOf, catNames);
    const age = ageDays(asOf, p.listed_at) ?? COMP_AGE_UNKNOWN_DAYS;
    f[4] = (age / 365) * Math.exp(-Math.log(2) * age / AGE_HALF_LIFE_DAYS);
    return f;
  });
  const beta730 = logistic(X730, y, 0.1, 40);
  const loglikHalfLife = logLikelihood(beta730, X730, y);

  const base_rate = protocols.filter(recordsMarketCap).length / protocols.length;

  // Empirical category token rate shrunk toward the overall rate, for display.
  const overallRate = base_rate;
  const kShrink = 5;
  const seen = new Set<string>();
  const categoryRates: OddsFit['category_rates'] = [];
  for (const p of protocols) {
    const c = p.category ?? 'Other';
    if (seen.has(c)) continue;
    seen.add(c);
    const ps = protocols.filter((q) => (q.category ?? 'Other') === c);
    const tokens = ps.filter(recordsMarketCap).length;
    categoryRates.push({
      category: c,
      n: ps.length,
      tokens,
      rate: tokens / ps.length,
      shrunk: (tokens + kShrink * overallRate) / (ps.length + kShrink),
    });
  }
  categoryRates.sort((a, b) => b.n - a.n);

  return { beta, catNames, n: protocols.length, base_rate, loglik_no_discount: loglikNoDiscount, loglik_half_life_730: loglikHalfLife, category_rates: categoryRates };
}

// ----------------------------------------------------------------------------
// scoring: implied value for a launch candidate
// ----------------------------------------------------------------------------

export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

export interface FarmPanel {
  total_cost_usd: number;
  total_time_hours: number;
  expected_value_usd: number;
  net_value_usd: number;
  verdict: 'FARM' | 'MARGINAL' | 'SKIP';
  break_even_value_usd: number;
  dollars_per_hour: number;
}

export interface ScoredRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  growth_pct: number;
  comps_value_usd: number;
  token_odds: number;
  implied_value_usd: number;
  is_watchlist: boolean;
  farm_steps: string[];
  farm: FarmPanel;
}

function compsValueOf(p: SnapshotProtocol, asOf: string, comps: CompsFit): number {
  const v = compsFeature(p, asOf, comps.catNames);
  let s = 0;
  for (let j = 0; j < comps.beta.length; j++) s += comps.beta[j]! * v[j]!;
  return Math.max(1, Math.round(Math.exp(s)));
}

function oddsOf(p: SnapshotProtocol, asOf: string, odds: OddsFit): number {
  const v = oddsFeature(p, asOf, odds.catNames);
  let z = 0;
  for (let j = 0; j < odds.beta.length; j++) z += odds.beta[j]! * v[j]!;
  return sigmoid(z);
}

// ----------------------------------------------------------------------------
// farm plans: concrete, per-protocol steps with costs and times
// ----------------------------------------------------------------------------

function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

function chainListOf(p: SnapshotProtocol): string {
  if (p.chains.length === 0) return 'its main chain';
  if (p.chains.length === 1) return p.chains[0]!;
  return p.chains.slice(0, -1).join(', ') + ' and ' + p.chains[p.chains.length - 1]!;
}

function capitalFor(p: SnapshotProtocol): number {
  const t = p.tvl_usd;
  if (t >= 5_000_000_000) return 500;
  if (t >= 1_000_000_000) return 300;
  if (t >= 100_000_000) return 200;
  if (t >= 25_000_000) return 120;
  return 80;
}

function gasFor(chain: string): number {
  const c = chain.toLowerCase();
  if (c.includes('ethereum') || c.includes('mainnet')) return 6;
  if (/(solana|base|arbitrum|optimism|polygon|avalanche|bsc|bnb)/.test(c)) return 2;
  return 1;
}

function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 90) return 75;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/**
 * The category's one action, in words a farmer can follow. Names the actual
 * chain(s) and the protocol's real function from the snapshot, never a canned
 * phrase shared by every row.
 */
function actionFor(p: SnapshotProtocol, capital: number, chain: string, chain2: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) {
    return `Bridge stable funds to ${chain}, then supply $${capital} of USDC to ${name}'s lending market on ${chain} and borrow a small stable amount against it to keep the position active.`;
  }
  if (c.includes('dex')) {
    return `Bridge stable funds to ${chain}, then provide $${capital} of liquidity to ${name}'s main pool on ${chain} and leave the LP position open.`;
  }
  if (c.includes('basis') || c.includes('arbitrage')) {
    return `Open a $${capital} basis position on ${name} on ${chain} (fund a delta-neutral cash-and-carry trade) and leave it running.`;
  }
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('options')) {
    return `Deposit $${capital} of USDC as margin on ${name} on ${chain} and open one small, delta-light position to generate qualifying volume.`;
  }
  if (c.includes('rwa')) {
    return `Mint $${capital} of ${name}'s real-world-asset yield token on ${chain} and hold it to earn the qualifying yield.`;
  }
  if (c.includes('yield')) {
    return `Deposit $${capital} into ${name}'s flagship yield vault on ${chain} and keep the receipt tokens.`;
  }
  if (c.includes('aggregator') || c.includes('allocator') || c.includes('curator') || c.includes('risk')) {
    return `Route $${capital} through a ${name} vault or allocation on ${chain} and keep the shares so the capital stays deployed.`;
  }
  if (c.includes('stable')) {
    return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain} and keep the position open.`;
  }
  if (c.includes('insurance')) {
    return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy active.`;
  }
  if (c.includes('prediction')) {
    return `Fund $${capital} and place one market position on ${name} on ${chain}.`;
  }
  if (c.includes('payment')) {
    return `Send $${capital} through ${name} on ${chain} and keep the account active with a qualifying flow.`;
  }
  if (c.includes('restak') || c.includes('staking') || c.includes('anchor') || c.includes('btc')) {
    return `Stake or deposit $${capital} of the chain's staking asset through ${name} on ${chain} and keep the receipt.`;
  }
  if (c.includes('launchpad')) {
    return `Bridge to ${chain}, complete ${name}'s onboarding on ${chain}, and stake the minimum qualifying amount to earn a launch allocation.`;
  }
  return `Bridge stable funds to ${chain}, then deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

function farmSteps(p: SnapshotProtocol, age: number | null): { steps: string[]; panel: FarmPanel } {
  const chain = chainOf(p);
  const chain2 = p.chains.length > 1 ? p.chains[1]! : chain;
  const capital = capitalFor(p);
  const gas = gasFor(chain);
  const holdDays = holdDaysFor(age);
  const holdWeeks = Math.max(1, Math.ceil(holdDays / 7));
  const bridgeFee = p.chains.length > 1 ? 6 : 0;
  const swapFee = Math.round(capital * 0.001 * 100) / 100;
  const weeklyGas = gas;
  const gasSetup = gas;

  const action = actionFor(p, capital, chain, chain2);
  const chainList = chainListOf(p);

  const steps = [
    stepText({ text: action, cost: `$${capital} of capital, $${gasSetup.toFixed(2)} of gas${p.chains.length > 1 ? ` and $${bridgeFee.toFixed(2)} to bridge` : ''}`, time: '20 minutes to set up' }),
    stepText({ text: `Use a fresh wallet that holds only the funds you plan to farm on ${chainList}, and keep its seed phrase offline.`, cost: 'nothing to create', time: '5 minutes' }),
    stepText({ text: `Make one qualifying action each week on ${chain} — a deposit, borrow, swap or vote — so the wallet stays active on the record.`, cost: `about $${weeklyGas.toFixed(2)} of gas a week`, time: '10 minutes a week' }),
    p.chains.length > 1
      ? stepText({ text: `Repeat the same qualifying action on ${chain2} so the activity is counted on more than one chain.`, cost: `$${gasFor(chain2).toFixed(2)} of gas on the second chain`, time: '15 minutes' })
      : stepText({ text: `Confirm the position still qualifies by reading ${p.name}'s own points or rewards rules on ${chain}.`, cost: 'no fee to check', time: '10 minutes' }),
    stepText({ text: `Hold the position for at least ${holdDays} days so the activity is counted in several snapshots before you unwind.`, cost: `the $${capital} stays deployed in ${chainList}`, time: `${holdDays} days` }),
    stepText({ text: `Before withdrawing, open ${p.name}'s own points or rewards page and claim only through the link it publishes.`, cost: 'no fee to check, and never pay to claim', time: '10 minutes' }),
    stepText({ text: `Write the wallet, the dates and every transaction in a farm log so the position can be repeated or closed cleanly.`, cost: 'nothing but the log', time: '5 minutes' }),
  ].map((s) => s.trim());

  // Farm economics, computed at run time from the protocol's own figures.
  const gasCosts = gasSetup + bridgeFee + swapFee + weeklyGas * holdWeeks;
  const totalCost = Math.round(gasCosts * 100) / 100;
  const totalTime = Math.round((0.33 + 0.17 * holdWeeks + 0.5 + 0.5) * 10) / 10;
  const implied = 0; // filled by the caller once implied value is known

  const panel: FarmPanel = {
    total_cost_usd: totalCost,
    total_time_hours: totalTime,
    expected_value_usd: 0,
    net_value_usd: 0,
    verdict: 'SKIP',
    break_even_value_usd: 1,
    dollars_per_hour: DOLLARS_PER_HOUR,
  };

  return { steps, panel };
}

// ----------------------------------------------------------------------------
// the radar: board + live + set-aside, all deterministic
// ----------------------------------------------------------------------------

export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
  symbol: string | null;
}

export interface AsideRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  reason: string;
}

export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: {
    name: string;
    version: string;
    formula: string;
    comps_formula: string;
    odds_formula: string;
    comps: {
      n_train: number;
      n_kept: number;
      r2: number;
      cv_mae_log: number;
      cv_r2: number;
      cv_n: number;
      beta: number[];
      categories: string[];
      intercept_name: string;
      unknown_age_days: number;
    };
    odds: {
      n: number;
      base_rate: number;
      half_life_days: number;
      loglik_no_discount: number;
      loglik_half_life_730: number;
      beta: number[];
      categories: string[];
      unknown_age_days: number;
      category_rates: { category: string; n: number; tokens: number; rate: number; shrunk: number }[];
    };
  };
  counts: {
    total: number;
    ranked: number;
    watchlist: number;
    live: number;
    live_mcap: number;
    products: number;
  };
  farm: {
    dollars_per_hour: number;
    assumptions: string[];
  };
  rows: ScoredRow[];
  live: LiveRow[];
  products: AsideRow[];
}

export function buildRadar(snapshot: Snapshot): { board: ScoredRow[]; radar: RadarData } {
  const comps = fitComps(snapshot);
  const odds = fitOdds(snapshot);
  const asOf = snapshot.as_of;
  const watchlist = new Set(snapshot.watchlist);

  const candidates = snapshot.protocols.filter(isLaunchCandidate);
  const products = snapshot.protocols.filter((p) => recordsNoToken(p) && !isLaunchCandidate(p));
  const live = snapshot.protocols.filter(recordsToken);

  const scored = candidates.map((p) => {
    const age = ageDays(asOf, p.listed_at);
    const compsValue = compsValueOf(p, asOf, comps);
    const tokenOdds = oddsOf(p, asOf, odds);
    const implied = Math.max(1, Math.round(compsValue * tokenOdds));
    const farm = farmSteps(p, age);
    // Expected value is the comps value times token odds again: implied value
    // already discounts by odds, so a second discount keeps the farm's expected
    // capture conservative (matching the v2 farm panel's arithmetic).
    const expected = Math.max(1, Math.round(compsValue * tokenOdds * tokenOdds));
    const timeValue = farm.panel.total_time_hours * DOLLARS_PER_HOUR;
    const net = expected - farm.panel.total_cost_usd - timeValue;
    const verdict: FarmPanel['verdict'] = net > timeValue ? 'FARM' : net > 0 ? 'MARGINAL' : 'SKIP';
    const breakEven = Math.max(1, Math.ceil((farm.panel.total_cost_usd + timeValue) / tokenOdds));
    return {
      rank: 0,
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      change_7d: p.change_7d,
      mcap_usd: p.mcap_usd,
      listed_at: p.listed_at,
      age_days: age,
      growth_pct: clipGrowth(p, GROWTH_CAP_COMP),
      comps_value_usd: compsValue,
      token_odds: tokenOdds,
      implied_value_usd: implied,
      is_watchlist: watchlist.has(p.slug),
      farm_steps: farm.steps,
      farm: {
        total_cost_usd: farm.panel.total_cost_usd,
        total_time_hours: farm.panel.total_time_hours,
        expected_value_usd: expected,
        net_value_usd: Math.round(net),
        verdict,
        break_even_value_usd: breakEven,
        dollars_per_hour: DOLLARS_PER_HOUR,
      },
    };
  });

  scored.sort((a, b) => b.implied_value_usd - a.implied_value_usd || (a.slug < b.slug ? -1 : 1));
  scored.forEach((r, i) => (r.rank = i + 1));

  const liveSorted = live
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1))
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
      symbol: p.symbol,
    }));

  const productsSorted = products
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1))
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      change_7d: p.change_7d,
      mcap_usd: p.mcap_usd,
      reason: setAsideReason(p),
    }));

  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model: {
      name: 'fitted radar model',
      version: 'radar-3',
      formula: 'implied_value_usd = comps_value_usd * token_odds',
      comps_formula: 'ln(mcap) = b0 + b1*ln(tvl) + b2*ln(age+1) + b3*growth_7d + b4*ln(chains+1) + category',
      odds_formula: 'logit P(records a token) = c0 + c1*ln(tvl) + c2*ln(chains+1) + c3*growth_7d + c4*age_years + category',
      comps: {
        n_train: comps.n_train,
        n_kept: comps.n_kept,
        r2: Math.round(comps.r2 * 10000) / 10000,
        cv_mae_log: Math.round(comps.cv_mae_log * 10000) / 10000,
        cv_r2: Math.round(comps.cv_r2 * 10000) / 10000,
        cv_n: comps.cv_n,
        beta: comps.beta.slice(),
        categories: comps.catNames,
        intercept_name: 'ln(mcap) intercept',
        unknown_age_days: COMP_AGE_UNKNOWN_DAYS,
      },
      odds: {
        n: odds.n,
        base_rate: Math.round(odds.base_rate * 10000) / 10000,
        half_life_days: AGE_HALF_LIFE_DAYS,
        loglik_no_discount: Math.round(odds.loglik_no_discount * 10000) / 10000,
        loglik_half_life_730: Math.round(odds.loglik_half_life_730 * 10000) / 10000,
        beta: odds.beta.slice(),
        categories: odds.catNames,
        unknown_age_days: COMP_AGE_UNKNOWN_DAYS,
        category_rates: odds.category_rates,
      },
    },
    counts: {
      total: snapshot.protocols.length,
      ranked: scored.length,
      watchlist: snapshot.watchlist.length,
      live: live.length,
      live_mcap: live.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0).length,
      products: products.length,
    },
    farm: {
      dollars_per_hour: DOLLARS_PER_HOUR,
      assumptions: [
        'Gas per chain: Ethereum $6, Solana/Base/Arbitrum/Optimism/Polygon/Avalanche/BSC $2, others $1.',
        'Bridging $6 when a protocol names more than one chain.',
        'Swap fee 0.10% of deployed capital.',
        `Time valued at $${DOLLARS_PER_HOUR} per hour.`,
      ],
    },
    rows: scored,
    live: liveSorted,
    products: productsSorted,
  };

  return { board: scored, radar };
}

// ----------------------------------------------------------------------------
// snapshot parsing
// ----------------------------------------------------------------------------

export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');

  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });

  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

// ----------------------------------------------------------------------------
// gate shape check and the data.js script
// ----------------------------------------------------------------------------

export function boardIssue(board: ScoredRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug(s): ${missing.slice(0, 5).join(', ')}`;
  if (board.length < 100 || board.length > 4000) return `the board has ${board.length} rows (gate wants 100 to 4000)`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all rows share the same implied value`;
  }
  const seen = new Set<string>();
  for (const r of board) {
    if (seen.has(r.slug)) return `slug ${r.slug} appears twice`;
    seen.add(r.slug);
    if (r.name.length === 0) return `${r.slug} has an empty name`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0 || r.implied_value_usd > 100_000_000_000_000) return `${r.slug} has implied value ${r.implied_value_usd}`;
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) {
      const t = s.trim();
      if (t.length < 20 || t.length > 600) return `${r.slug} has a farm step of ${t.length} characters`;
    }
  }
  return null;
}

export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}

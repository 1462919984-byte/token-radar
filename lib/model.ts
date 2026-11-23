// token-radar-01 v2: a fitted value model, stdlib only, deterministic.
//
// Implied value = fitted comps value x token odds, both learned from the snapshot's
// own figures. The comps fit is a Huber-robust ridge regression of log market cap on
// log TVL, log(age+1), 7-day growth, log(chain count + 1) and the seven largest
// category indicators. The token-odds fit is a logistic regression over all 758
// protocols on the same features with age entering by its own sign (no half-life
// discount; the record does not support one).

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

export interface FarmStep { text: string; cost: string; time: string; }

export interface BoardRow {
  rank: number;
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
  token_status: 'pre-token';
  launch_candidate: boolean;
  age_days: number | null;
  implied_value_usd: number;
  comps_value_usd: number;
  token_odds: number;
  farm_steps: string[];
  farm: {
    total_cost_usd: number;
    total_time_hours: number;
    expected_value_usd: number;
    net_value_usd: number;
    verdict: 'FARM' | 'MARGINAL' | 'SKIP';
    break_even_value_usd: number;
    dollars_per_hour: number;
  };
}

export interface RadarRow {
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
  launch_candidate: boolean;
  implied_value_usd: number;
  comps_value_usd: number;
  token_odds: number;
  farm_steps: string[];
  farm: {
    total_cost_usd: number;
    total_time_hours: number;
    expected_value_usd: number;
    net_value_usd: number;
    verdict: 'FARM' | 'MARGINAL' | 'SKIP';
    break_even_value_usd: number;
    dollars_per_hour: number;
  };
}

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

export interface FitStats {
  full_r2: number;
  cv5_mae: number;
  walk_jun_mae: number;
  walk_jan_mae: number;
  loo_median_abs: number;
  loo_implied_ratio: number;
  n_comps: number;
  med_age_days: number;
  n_train_jun: number;
  n_test_jun: number;
  n_train_jan: number;
  n_test_jan: number;
  loglik_no_discount: number;
  loglik_half_life_730: number;
  category_rates: { category: string; n: number; tokens: number; rate: number; shrunk: number }[];
}

export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  pre_token_total: number;
  rows: RadarRow[];
  live: LiveRow[];
  fit: FitStats;
  fit_model: { [key: string]: number };
  odds_model: { [key: string]: number };
  coverage: { rows: number; watchlist: number };
}

export class SnapshotError extends Error {
  constructor(message: string) { super(message); this.name = 'SnapshotError'; }
}

const DAY_MS = 86_400_000;
export const MODEL_VERSION = 'radar-2';

// ---- top categories for both fits: the seven largest among the 128 comps ----
const TOP_CATEGORIES = ['Lending', 'Dexs', 'Liquid Staking', 'Yield', 'CEX', 'RWA', 'Canonical Bridge'];

function finite(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

export function hasTokenRecord(p: SnapshotProtocol): boolean {
  return p.mcap_usd !== null;
}

const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const EXCHANGE = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/i;

export function isWrappedOrStaked(p: SnapshotProtocol): boolean {
  const n = p.name.toLowerCase();
  const excluded = ['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX'];
  return /\b(wrapped|staked|staking)\b/.test(n) || excluded.includes(p.category ?? '');
}

export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name) && !isWrappedOrStaked(p) && !EXCHANGE.test(p.name);
}

function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

// ---------------- linear algebra helpers (deterministic) ----------------
function median(values: number[]): number {
  const a = [...values].sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m]! : (a[m - 1]! + a[m]!) / 2;
}

function ridgeSolve(X: number[][], y: number[], k: number, lambda: number, w: number[] | null): Float64Array {
  const A = Array.from({ length: k }, () => new Float64Array(k));
  const b = new Float64Array(k);
  for (let i = 0; i < X.length; i++) {
    const wi = w ? w[i]! : 1;
    const xi = X[i]!;
    for (let a = 0; a < k; a++) {
      b[a] += wi * xi[a]! * y[i]!;
      for (let c = 0; c < k; c++) A[a]![c] += wi * xi[a]! * xi[c]!;
    }
  }
  for (let a = 1; a < k; a++) A[a]![a] += lambda;
  for (let a = 0; a < k; a++) A[a]![a] += 1e-9;
  const M = A.map((row, j) => {
    const r = new Float64Array(k + 1);
    for (let c = 0; c < k; c++) r[c] = row[c]!;
    r[k] = b[j]!;
    return r;
  });
  for (let col = 0; col < k; col++) {
    let piv = col;
    let best = Math.abs(M[col]![col]!);
    for (let r = col + 1; r < k; r++) {
      if (Math.abs(M[r]![col]!) > best) { best = Math.abs(M[r]![col]!); piv = r; }
    }
    if (best < 1e-10) continue;
    [M[col], M[piv]] = [M[piv]!, M[col]!];
    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const f = M[r]![col]! / M[col]![col]!;
      for (let c = col; c <= k; c++) M[r]![c] -= f * M[col]![c]!;
    }
  }
  const beta = new Float64Array(k);
  for (let j = 0; j < k; j++) beta[j] = M[j]![k]! / M[j]![j]!;
  return beta;
}

function huberRidge(X: number[][], y: number[], k: number, lambda: number, tune = 1.345, iters = 60): Float64Array {
  let beta = ridgeSolve(X, y, k, lambda, null);
  const n = y.length;
  for (let it = 0; it < iters; it++) {
    const r = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      let pr = 0;
      for (let j = 0; j < k; j++) pr += beta[j]! * X[i]![j]!;
      r[i] = y[i]! - pr;
    }
    const sigma = 1.4826 * median(r.map((v) => Math.abs(v)));
    const eps = sigma < 1e-9 ? 1e-9 : sigma;
    const w = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const z = Math.abs(r[i]!) / eps;
      w[i] = z <= tune ? 1 : tune / z;
    }
    const nb = ridgeSolve(X, y, k, lambda, w);
    let d = 0;
    for (let j = 0; j < k; j++) d = Math.max(d, Math.abs(nb[j]! - beta[j]!));
    beta = nb;
    if (d < 1e-7) break;
  }
  return beta;
}

function logitFit(X: number[][], y: number[], k: number, lambda = 0.1, iters = 60): Float64Array {
  let beta = new Float64Array(k);
  const n = y.length;
  for (let it = 0; it < iters; it++) {
    const w = new Array<number>(n);
    const z = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < k; j++) s += beta[j]! * X[i]![j]!;
      const p = 1 / (1 + Math.exp(-s));
      const e = 1e-6;
      const p2 = Math.min(Math.max(p, e), 1 - e);
      w[i] = p2 * (1 - p2);
      z[i] = s + (y[i]! - p2) / w[i]!;
    }
    const A = Array.from({ length: k }, () => new Float64Array(k));
    const b = new Float64Array(k);
    for (let i = 0; i < n; i++) {
      const wi = w[i]!;
      for (let a = 0; a < k; a++) {
        b[a] += wi * X[i]![a]! * z[i]!;
        for (let c = 0; c < k; c++) A[a]![c] += wi * X[i]![a]! * X[i]![c]!;
      }
    }
    for (let a = 1; a < k; a++) A[a]![a] += lambda;
    for (let a = 0; a < k; a++) A[a]![a] += 1e-9;
    const M = A.map((row, j) => {
      const r = new Float64Array(k + 1);
      for (let c = 0; c < k; c++) r[c] = row[c]!;
      r[k] = b[j]!;
      return r;
    });
    for (let col = 0; col < k; col++) {
      let piv = col;
      let best = Math.abs(M[col]![col]!);
      for (let r = col + 1; r < k; r++) {
        if (Math.abs(M[r]![col]!) > best) { best = Math.abs(M[r]![col]!); piv = r; }
      }
      if (best < 1e-10) continue;
      [M[col], M[piv]] = [M[piv]!, M[col]!];
      for (let r = 0; r < k; r++) {
        if (r === col) continue;
        const f = M[r]![col]! / M[col]![col]!;
        for (let c = col; c <= k; c++) M[r]![c] -= f * M[col]![c]!;
      }
    }
    const nb = new Float64Array(k);
    for (let j = 0; j < k; j++) nb[j] = M[j]![k]! / M[j]![j]!;
    let d = 0;
    for (let j = 0; j < k; j++) d = Math.max(d, Math.abs(nb[j]! - beta[j]!));
    beta = nb;
    if (d < 1e-8) break;
  }
  return beta;
}

function predict(beta: Float64Array, x: number[]): number {
  let s = 0;
  for (let j = 0; j < beta.length; j++) s += beta[j]! * x[j]!;
  return s;
}

// ---------------- feature construction ----------------
function featRow(p: SnapshotProtocol, asOf: string, medAge: number): number[] {
  const age = ageDays(asOf, p.listed_at) ?? medAge;
  const lt = Math.log(p.tvl_usd);
  const la = Math.log(age + 1);
  const g = Math.min(Math.max(finite(p.change_7d, 0), -60), 120) / 100;
  const lc = Math.log(p.chains.length + 1);
  const cats = TOP_CATEGORIES.map((c) => ((p.category ?? 'Other') === c ? 1 : 0));
  return [1, lt, la, g, lc, ...cats];
}

function designComps(protocols: SnapshotProtocol[], asOf: string, medAge: number): { X: number[][]; y: number[]; k: number } {
  const X = protocols.map((p) => featRow(p, asOf, medAge));
  const y = protocols.map((p) => Math.log(p.mcap_usd!));
  return { X, y, k: 1 + 4 + TOP_CATEGORIES.length };
}

function designOdds(protocols: SnapshotProtocol[], asOf: string, medAge: number): { X: number[][]; y: number[]; k: number } {
  const X = protocols.map((p) => {
    const f = featRow(p, asOf, medAge);
    // age enters with its own sign, not as log(age+1): replace feature index 2
    f[2] = (ageDays(asOf, p.listed_at) ?? medAge) / 365;
    return f;
  });
  const y = protocols.map((p) => (hasTokenRecord(p) ? 1 : 0));
  return { X, y, k: 1 + 4 + TOP_CATEGORIES.length };
}

// ---------------- model fitting ----------------
function fitModels(snapshot: Snapshot): {
  compsBeta: Float64Array;
  oddsBeta: Float64Array;
  medAge: number;
  stats: FitStats;
  compsMap: { [k: string]: number };
  oddsMap: { [k: string]: number };
} {
  const asOf = snapshot.as_of;
  const all = snapshot.protocols;
  const comps = all.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0);
  const medAge = median(comps.filter((p) => p.listed_at !== null).map((p) => ageDays(asOf, p.listed_at)!));

  const { X: Xc, y: yc, k: kc } = designComps(comps, asOf, medAge);
  const compsBeta = huberRidge(Xc, yc, kc, 3);
  const { X: Xo, y: yo, k: ko } = designOdds(all, asOf, medAge);
  const oddsBeta = logitFit(Xo, yo, ko);

  // R-squared on full 128
  const meanY = yc.reduce((a, b) => a + b, 0) / yc.length;
  const ssTot = yc.reduce((a, b) => a + (b - meanY) ** 2, 0);
  const ssRes = yc.reduce((a, b, i) => a + (b - predict(compsBeta, Xc[i]!)) ** 2, 0);
  const fullR2 = 1 - ssRes / ssTot;

  // 5-fold CV (deterministic index-based folds)
  const n = comps.length;
  const folds = 5;
  let cv5Mae = 0;
  for (let f = 0; f < folds; f++) {
    const tr = comps.filter((_, i) => i % folds !== f);
    const te = comps.filter((_, i) => i % folds === f);
    const { X: Xtr, y: ytr, k: ktr } = designComps(tr, asOf, medAge);
    const beta = huberRidge(Xtr, ytr, ktr, 3);
    const { X: Xte, y: yte } = designComps(te, asOf, medAge);
    for (let i = 0; i < te.length; i++) cv5Mae += Math.abs(predict(beta, Xte[i]!) - yte[i]!);
  }
  cv5Mae /= n;

  // walk-forward
  function walkForward(cutoff: string): { mae: number; tr: number; te: number } {
    const cutoffMs = Date.parse(`${cutoff}T00:00:00Z`);
    const tr = comps.filter((p) => p.listed_at !== null && Date.parse(`${p.listed_at}T00:00:00Z`) < cutoffMs);
    const te = comps.filter((p) => p.listed_at !== null && Date.parse(`${p.listed_at}T00:00:00Z`) >= cutoffMs);
    const { X: Xtr, y: ytr, k: ktr } = designComps(tr, asOf, medAge);
    const beta = huberRidge(Xtr, ytr, ktr, 3);
    const { X: Xte, y: yte } = designComps(te, asOf, medAge);
    let mae = 0;
    for (let i = 0; i < te.length; i++) mae += Math.abs(predict(beta, Xte[i]!) - yte[i]!);
    return { mae: te.length ? mae / te.length : 0, tr: tr.length, te: te.length };
  }
  const walkJun = walkForward('2025-06-01');
  const walkJan = walkForward('2025-01-01');

  // leave-one-out median absolute error
  const looAbs: number[] = [];
  for (let i = 0; i < n; i++) {
    const tr = comps.filter((_, j) => j !== i);
    const { X: Xtr, y: ytr, k: ktr } = designComps(tr, asOf, medAge);
    const beta = huberRidge(Xtr, ytr, ktr, 3);
    const x = designComps([comps[i]!], asOf, medAge).X[0]!;
    looAbs.push(Math.abs(predict(beta, x) - Math.log(comps[i]!.mcap_usd!)));
  }
  const looMedian = median(looAbs);
  const looRatio = Math.exp(looMedian);

  // logistic log-likelihood: no discount vs 730-day half-life
  function logLik(beta: Float64Array, X: number[][], y: number[]): number {
    let s = 0;
    for (let i = 0; i < y.length; i++) {
      const z = predict(beta, X[i]!);
      s += y[i]! * z - Math.log(1 + Math.exp(z));
    }
    return s;
  }
  const loglikNoDiscount = logLik(oddsBeta, Xo, yo);
  // 730-day half-life: refit with discounted age feature
  const { X: Xo730, k: ko730 } = (() => {
    const X = all.map((p) => {
      const f = featRow(p, asOf, medAge);
      const a = ageDays(asOf, p.listed_at) ?? medAge;
      f[2] = (a / 365) * Math.exp(-Math.log(2) * a / 730);
      return f;
    });
    return { X, k: 1 + 4 + TOP_CATEGORIES.length };
  })();
  const oddsBeta730 = logitFit(Xo730, yo, ko730);
  const loglikHalfLife = logLik(oddsBeta730, Xo730, yo);

  // category rates (empirical + shrunk toward overall)
  const overallRate = all.filter(hasTokenRecord).length / all.length;
  const categoryRates: FitStats['category_rates'] = [];
  const seen = new Set<string>();
  for (const p of all) {
    const c = p.category ?? 'Other';
    if (seen.has(c)) continue;
    seen.add(c);
    const ps = all.filter((q) => (q.category ?? 'Other') === c);
    const tokens = ps.filter(hasTokenRecord).length;
    const kShrink = 5;
    categoryRates.push({
      category: c,
      n: ps.length,
      tokens,
      rate: tokens / ps.length,
      shrunk: (tokens + kShrink * overallRate) / (ps.length + kShrink),
    });
  }
  categoryRates.sort((a, b) => b.n - a.n);

  const stats: FitStats = {
    full_r2: fullR2,
    cv5_mae: cv5Mae,
    walk_jun_mae: walkJun.mae,
    walk_jan_mae: walkJan.mae,
    loo_median_abs: looMedian,
    loo_implied_ratio: looRatio,
    n_comps: n,
    med_age_days: medAge,
    n_train_jun: walkJun.tr,
    n_test_jun: walkJun.te,
    n_train_jan: walkJan.tr,
    n_test_jan: walkJan.te,
    loglik_no_discount: loglikNoDiscount,
    loglik_half_life_730: loglikHalfLife,
    category_rates: categoryRates,
  };

  const compsMap: { [k: string]: number } = { intercept: compsBeta[0]! };
  const oddsMap: { [k: string]: number } = { intercept: oddsBeta[0]! };
  const compsNames = ['log_tvl', 'log_age_plus_one', 'growth_7d', 'log_chains_plus_one', ...TOP_CATEGORIES];
  const oddsNames = ['log_tvl', 'log_chains_plus_one', 'growth_7d', 'age_years', ...TOP_CATEGORIES];
  for (let j = 1; j < kc; j++) compsMap[compsNames[j - 1]!] = compsBeta[j]!;
  for (let j = 1; j < ko; j++) oddsMap[oddsNames[j - 1]!] = oddsBeta[j]!;

  return { compsBeta, oddsBeta, medAge, stats, compsMap, oddsMap };
}

// ---------------- implied value ----------------
function compsValue(p: SnapshotProtocol, asOf: string, medAge: number, beta: Float64Array): number {
  const x = featRow(p, asOf, medAge);
  const logVal = predict(beta, x);
  return Math.max(1, Math.exp(logVal));
}

function tokenOdds(p: SnapshotProtocol, asOf: string, medAge: number, beta: Float64Array): number {
  const f = featRow(p, asOf, medAge);
  f[2] = (ageDays(asOf, p.listed_at) ?? medAge) / 365;
  const z = predict(beta, f);
  return 1 / (1 + Math.exp(-z));
}

// ---------------- farm plan ----------------
const DOLLARS_PER_HOUR = 40;

function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

function gasPerChain(chain: string): number {
  const c = chain.toLowerCase();
  if (c.includes('ethereum') || c.includes('mainnet')) return 6;
  if (c.includes('solana') || c.includes('base') || c.includes('arbitrum') || c.includes('optimism') || c.includes('polygon') || c.includes('avalanche') || c.includes('bsc') || c.includes('bnb')) return 2;
  return 1;
}

function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 500;
  if (p.tvl_usd >= 1_000_000_000) return 300;
  if (p.tvl_usd >= 100_000_000) return 200;
  return 100;
}

function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chain} and keep the position open.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} and keep the position open.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chain} through ${name} and use the bridged funds in its app.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's staking asset through ${name} and keep the receipt.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${name} and open one small hedge.`;
  if (c.includes('prediction')) return `Fund $${capital} and place one market position on ${name}.`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`;
  if (c.includes('rwa')) return `Mint $${capital} of the protocol's yield asset on ${chain} and hold it.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

export function farmSteps(p: SnapshotProtocol, age: number | null): { steps: string[]; totalCost: number; totalTime: number } {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = gasPerChain(chain);
  const bridgeFee = p.chains.length > 1 ? 6 : 0;
  const swapFee = Math.round(capital * 0.001 * 100) / 100;
  const gasSteps = gas + bridgeFee + swapFee + 4;
  const totalCost = Math.round(gasSteps * 100) / 100;
  const totalTime = 0.15 + 0.3 + 0.7 + hold * 0.25 + 0.2 + 0.1;
  const steps = [
    { text: `Use a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline.`, cost: `$${gas.toFixed(2)} of gas for the first transaction`, time: '5 minutes' },
    { text: actionFor(p, capital, chain), cost: `$${capital} of capital, $${gas.toFixed(2)} of gas${p.chains.length > 1 ? `, $${bridgeFee.toFixed(2)} to bridge` : ''}`, time: '15 minutes to set up' },
    { text: `Swap or deposit once a week to keep activity on the record, such as a deposit, a borrow, a swap or a vote.`, cost: `about $${gas.toFixed(2)} of gas a week`, time: '10 minutes a week' },
    { text: `Hold the position for at least ${hold} days so the activity is counted in more than one snapshot.`, cost: `the $${capital} stays in the position`, time: `${hold} days` },
    { text: `Before you withdraw, open the protocol's own points or rewards page and claim only from the link it publishes.`, cost: 'no fee to check, and never pay to claim', time: '10 minutes' },
    { text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`, cost: 'nothing but the log', time: '5 minutes' },
  ].map(stepText);
  return { steps, totalCost, totalTime: Math.round(totalTime * 10) / 10 };
}

function roundToWhole(n: number): number {
  return Math.max(1, Math.round(n));
}

// ---------------- build board ----------------
export function buildBoard(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const { compsBeta, oddsBeta, medAge, stats, compsMap, oddsMap } = fitModels(snapshot);
  const index = bySlug(snapshot.protocols);
  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  // early radar: every pre-token protocol (no token on its own record), including
  // CEX, wrapped, staked and versioned pools whose parent token trades; they are
  // tagged launch_candidate=false but still ranked by the same implied-value model.
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isPreToken(p))
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0));
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    const cv = compsValue(p, snapshot.as_of, medAge, compsBeta);
    const odds = tokenOdds(p, snapshot.as_of, medAge, oddsBeta);
    const implied = roundToWhole(cv * odds);
    const age = ageDays(snapshot.as_of, p.listed_at);
    const farm = farmSteps(p, age);
    // Expected value is the implied value times token odds again: a conservative
    // expected capture for the farm, since implied value already discounts by odds.
    const expected = Math.round(cv * odds * odds);
    const timeValue = farm.totalTime * DOLLARS_PER_HOUR;
    const net = expected - farm.totalCost - timeValue;
    // Break-even implied value: the board value at which net equals zero.
    const breakEven = Math.max(1, Math.ceil((farm.totalCost + timeValue) / odds));
    let verdict: 'FARM' | 'MARGINAL' | 'SKIP';
    if (net > timeValue) verdict = 'FARM';
    else if (net > 0) verdict = 'MARGINAL';
    else verdict = 'SKIP';
    return { p, implied, cv: roundToWhole(cv), odds, age, farm, expected, net, breakEven, verdict };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));

  const board: BoardRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token',
    launch_candidate: isRankable(s.p),
    age_days: s.age,
    implied_value_usd: s.implied,
    comps_value_usd: s.cv,
    token_odds: s.odds,
    farm_steps: s.farm.steps,
    farm: {
      total_cost_usd: s.farm.totalCost,
      total_time_hours: s.farm.totalTime,
      expected_value_usd: Math.round(s.expected),
      net_value_usd: Math.round(s.net),
      verdict: s.verdict,
      break_even_value_usd: s.breakEven,
      dollars_per_hour: DOLLARS_PER_HOUR,
    },
  }));

  const live = snapshot.protocols
    .filter((p) => !isPreToken(p))
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
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

  const early_size = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
  const pre_token_total = snapshot.protocols.filter(isPreToken).length;

  const rows: RadarRow[] = board.map((r) => ({
    rank: r.rank,
    slug: r.slug,
    name: r.name,
    category: r.category,
    chains: r.chains,
    tvl_usd: r.tvl_usd,
    change_7d: r.change_7d,
    mcap_usd: r.mcap_usd,
    listed_at: r.listed_at,
    age_days: r.age_days,
    launch_candidate: r.launch_candidate,
    implied_value_usd: r.implied_value_usd,
    comps_value_usd: r.comps_value_usd,
    token_odds: r.token_odds,
    farm_steps: r.farm_steps,
    farm: r.farm,
  }));

  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size,
    live_size: live.length,
    pre_token_total,
    rows,
    live,
    fit: stats,
    fit_model: compsMap,
    odds_model: oddsMap,
    coverage: { rows: board.length, watchlist: snapshot.watchlist.length },
  };

  return { board, radar };
}

export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try { raw = JSON.parse(text); } catch { throw new SnapshotError('snapshot.json is not JSON'); }
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
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}`;
  }
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}

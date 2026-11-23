# token-radar-01 (v3)

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/1462919984-byte/token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/1462919984-byte/token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

A deterministic Node 22 token launch radar. It reads the sealed DefiLlama protocol
snapshot, fits a value model at run time from the snapshot's own figures, and
ranks pre-token launch candidates by implied value.

## The model (in words)

- **Comps curve** — ordinary least squares of `ln(mcap)` on `ln(tvl)`,
  `ln(age_days + 1)`, 7-day growth, `ln(chains + 1)` and category indicators,
  fit on the 128 positive market caps, with two rounds of dropping residuals
  beyond 2.5 standard deviations. Protocols with no `listed_at` are imputed the
  median age of the token-bearing set (about 1,000 days).
- **Token odds** — a logistic fit for whether a protocol records a market cap,
  on all 758 protocols, with `ln(tvl)`, `ln(chains + 1)`, 7-day growth, age in
  years and category indicators. Age enters with its own fitted sign; the
  730-day half-life discount was tested against the data and removed.
- **Implied value** = `round(comps_value_usd * token_odds)`.

The model is fitted at run time — no protocol name is hardcoded anywhere.

## What it ships

- `board.json` — 197 ranked launch candidates, watchlist 40/40 covered, no
  derivative/CEX/bridge/wrapped/versioned pool in the ranking.
- A static, no-network dashboard (board, detail view, filter) with a farm
  economics panel for every row: cost, time, expected value, net value, verdict
  and break-even implied value.
- `dossier.json` written by the vendored `buildDossier`.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input input/snapshot.json
```

Two runs write byte-identical output: no wall clock, no randomness, no network,
no unordered iteration.
<sub>Entered on [make.fun](https://make.fun/?ref=nbsk8k44&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>

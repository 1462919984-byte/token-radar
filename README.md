# token-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/1462919984-byte/token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/1462919984-byte/token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Token launch radar: a deterministic Node 22 program that reads the sealed DefiLlama snapshot and ranks pre-token protocols by implied value. The value model is fitted at run time from the snapshot's own data — a comps curve (log market cap on log TVL, age, growth and chains) plus token odds from a logistic fit discounted by tokenless age — never a hardcoded list of names. Every ranked protocol carries concrete farm steps with costs and times, and the entry ships a static, no-network dashboard (board, detail view, filter) with screenshots.

## Run it

```sh
node main.ts --input <snapshot.json> --output <dir>
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=nbsk8k44&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>

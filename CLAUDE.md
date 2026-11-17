# CLAUDE.md — MVO on Kalshi binaries

A personal research project for learning mean-variance optimization (MVO) properly, by applying it to
NFL Kalshi contracts and seeing where it works and where it breaks. Kelly (log-utility) sizing is the
benchmark it gets compared against.

The owner has a quant background but no formal coursework in portfolio optimization. **Understanding
comes first; the code exists to make the ideas concrete.** It should also be clean enough to show
someone.

Reference: Paleologo, *Basic and Advanced Portfolio Optimization* (deck), and *Elements of
Quantitative Investing*. Slide numbers below refer to the deck.

---

## How to work in this repo

- **Explain before implementing.** Before writing the core function of a phase, write the derivation
  in `docs/notes/phase-N.md`. Keep it short, with each step checkable.
- **Label claims.** In notes, write-ups and explanations, tag statements as **[rigorous]** (follows
  from the math or is a standard result you can cite) or **[intuition]** (a useful picture, not a
  proof). Never present intuition as a result.
- **Plain language.** Short sentences, concrete numbers, no filler. If a sentence would survive being
  deleted, delete it.
- **Small, verifiable steps.** Every formula gets a unit test against a hand-computed case. Every
  simulation result gets a sanity check (e.g. the analytic mean vs. the simulated mean).
- **Ask, don't guess,** when a modeling choice changes the conclusions (which prices to use, which
  "truth" to evaluate under, how to treat fees). List the options and the trade-off.
- **No trading.** This repo never places orders, never imports `trader/` from the NFL repo, and never
  writes to the NFL server's paths. Research only.

---

## Upstream: the `nfl-playoff-algo` repo (read-only)

This project consumes the NFL model's outputs. Facts as of 2026-11-15:

- **Model:** team ratings from game lines → season Monte Carlo with official tiebreakers → daily fair
  values. 20k sims at each of three uncertainty scales (x1, x1.5, x2). The empirically chosen scale is x1.5.
- **Exported today:** daily CSV at 14:10 UTC,
  `/var/lib/nfl-collector/fairvalues/<YYYYMMDDTHHMMZ>-calibrated-hl2-k.csv`
  with `team, rating, p_playoffs (+se, band), p_division (+band), Kalshi bid/ask at T, sigma, tau,
  half_life, sims, slot`. There is no `settings` column: seed, nflverse file and code version are not
  recorded anywhere.
- **Not exported:** per-sim outcomes (the joint distribution), per-game home-win probabilities, and
  seed distributions. **Phase 3 needs joint per-sim outcomes** (see Phase 0).
- **Traded market:** `KXNFLPLAYOFF-27`, 32 contracts ("team makes playoffs"). Ticker mapping:
  `model.league.playoff_ticker(team)` → `KXNFLPLAYOFF-27-<code>` (JAX→JAC, LA→LAR). Division contracts
  are priced but not traded (maker fees). Post-only limit orders; no maker fee on the playoff series.
- **Prices:** `tidy.sqlite` on the server: order books every 30 min since 2026-11-02 ~02:30 UTC, and
  futures trades backfilled to 2026-02-10.
- **Point-in-time caveat:** fair values for Nov 2–5 were recomputed later with newer settings, so they
  are **not** point-in-time (they came from a one-off `--since` backfill). Only use files generated
  on or after 2026-11-06 14:10 UTC for anything that claims to be a historical decision, and verify
  each against its file timestamp and its `tau`/`half_life`/`sims` columns.
- **Current prod sizing:** fixed 1 contract per order with a $20 cap, and no Kelly. This is the baseline
  rule to compare against.
- **Settlement:** about 2027-01-11. Until then there are no realized outcomes for this season.
- **Known model weaknesses:** the x1.5 uncertainty scale is empirical. Simulated margins are too narrow
  (SD 11.4 vs 12.6 real). There are no injuries or Week 18 resting, and updates are once daily. Edge
  evidence is thin (+4.7¢ ± 1.7 markout, 22 teams, 2 weeks). **Treat model probabilities as noisy
  estimates, not truth.** That is the point of Phase 4.

---

## Math reference

### MVO (deck slides 3–6) [rigorous]
Assets with expected PnL vector μ and covariance Ω. Choose positions n to maximize

    nᵀμ − λ · nᵀΩn      →      n* = Ω⁻¹μ / (2λ)     (when Ω is invertible)

- λ only scales n*. It doesn't change direction. The direction Ω⁻¹μ maximizes the Sharpe ratio, and
  that maximum Sharpe is sqrt(μᵀΩ⁻¹μ).
- MVO is exactly optimal for (a) jointly normal PnL with exponential utility, or (b) quadratic utility.
  Otherwise it is a second-order approximation of utility and ignores skew and tails.

### Binary contracts, in contract units [rigorous]
Payoff Xᵢ ∈ {0, 1}, with model probability qᵢ = P(Xᵢ = 1).

- Buy YES at ask aᵢ: PnL per contract = Xᵢ − aᵢ, mean qᵢ − aᵢ.
- Buy NO at (1 − bidᵢ): PnL per contract = bidᵢ − Xᵢ, mean bidᵢ − qᵢ. So NO is a short YES at the bid.
- Covariance: Cov(Xᵢ, Xⱼ) = P(Xᵢ = 1, Xⱼ = 1) − qᵢqⱼ. Variance qᵢ(1 − qᵢ).
- If A ⊂ B (e.g. "wins division" ⊂ "makes playoffs"): Cov = P(A)(1 − P(B)).
- Because the bid ≠ the ask, positions are modeled as n⁺ ≥ 0 (YES) and n⁻ ≥ 0 (NO) per contract. The
  mean is then piecewise linear, and the problem is a convex QP, not a closed form.

### Single contract, per-dollar returns [rigorous]
Stake a fraction f of wealth on YES at price p, with belief q:
- return = (1 − p)/p with probability q, and −1 otherwise. Mean (q − p)/p. Variance q(1 − q)/p².
- MVO (λ = ½): f = (q − p)·p / (q(1 − q)).
- Kelly (maximize E[log wealth]): f = (q − p)/(1 − p).
- Check values: (p, q) = (0.50, 0.60) → Kelly 0.200, MVO 0.208. (0.50, 0.80) → 0.600, 0.938.
  (0.05, 0.10) → 0.053, 0.028.

### The playoff book is singular [rigorous]
Exactly 7 of 16 teams per conference make the playoffs, so Σ_{i∈AFC} Xᵢ = 7 in every outcome.
Therefore Var(Σ Xᵢ) = 0, which means 1ᵀΩ_AFC 1 = 0 and **Ω is singular** (the same holds for the NFC).

- Buying all 16 AFC YES contracts costs Σ asks and pays exactly 7: a riskless bond.
- Σ asks > 7 is the vig. Σ asks < 7 (or Σ bids > 7) would be an arbitrage, and MVO would go unbounded.
- Ω⁻¹ does not exist, so use the bid/ask QP (bounded by the spread) or a pseudo-inverse, and explain
  which one and why. Simulated Ω will have this property up to sampling error. Check it.

### Multi-asset Kelly [rigorous objective, numerical solution]
Maximize (1/S) Σₛ log(W₀ + Σᵢ (n⁺ᵢ(Xᵢˢ − aᵢ) + n⁻ᵢ(bidᵢ − Xᵢˢ))) over S simulated brackets.
This is concave in (n⁺, n⁻), so solve it with cvxpy, or with scipy if cvxpy is a hassle. There is no
closed form.

---

## Phases

Each phase ends with a short write-up in `docs/writeups/phase-N.md`: the question, the result (a plot
or table), what it means [tagged], and what surprised you.

### Phase 0 — Data contract
- Agree on a **joint-outcome export** from the NFL repo: for a given fair-value run, an `S × 32` 0/1
  matrix of "made playoffs" per sim (optionally `S × 32` division winners), plus the run's timestamp,
  settings and uncertainty scale. Format: `.npz` or parquet.
- Preferred approach: a small, **additive** export script in the NFL repo that doesn't change prod
  behavior. Fallback: import `model/` read-only from this repo and rerun the sim locally with the same
  settings. Confirm the probabilities reproduce the CSV within the standard error.
- Sync script: copy fair-value CSVs, joint exports, and the needed `tidy.sqlite` tables into `data/raw/`
  (gitignored). Never write to the server.
- Loader with validation: 32 teams, tickers map, every sim has exactly 7 per conference, and per-sim
  marginals match `p_playoffs`.

### Phase 1 — MVO mechanics
- Implement `mvo_weights(mu, Omega, lam)` and `max_sharpe(mu, Omega)`.
- Reproduce deck slide 5 exactly: α = (0.12, 0.08, 0.04), σ = (0.25, 0.20, 0.15), diagonal Ω, vol
  budget 0.10 → SR ≈ 0.679, w ≈ (0.283, 0.294, 0.262).
- Two-asset demo: sweep the correlation and show when MVO shorts a positive-alpha asset (deck "Shorting
  a Positive-Sharpe Asset").
- **Learn:** the closed form, what λ does, why Ω⁻¹ (not Ω) appears.

### Phase 2 — One contract: MVO vs Kelly
- Heatmap of f_MVO / f_Kelly over a (p, q) grid. Reproduce the check values above.
- Simulate many repeated bets with each rule. Compare log growth, final-wealth distribution, and P(ruin
  or −50%).
- Show that MVO = Kelly to second order: Taylor-expand log(1 + fR).
- **Learn:** exactly what the mean-variance approximation throws away (skew), and where it matters
  (big edges, longshots).

### Phase 3 — The 32-contract playoff book
- Build μ (from q and bid/ask) and Ω (from joint sims) for a chosen snapshot.
- Show the singularity numerically (eigenvalues of Ω_AFC, Ω_NFC) and connect it to the riskless
  "buy all 16" bond. Compute the vig per conference over time from `tidy.sqlite`.
- Solve: (a) the MVO QP with n⁺/n⁻, (b) multi-asset Kelly over sims, (c) independent per-contract Kelly
  (ignores correlation), and (d) the prod baseline (1 contract per edge, $20 cap).
- Compare positions side by side. Where does using Ω create hedges (e.g. NO on a conference rival)?
- **Learn:** what the covariance matrix actually does, and why correlation-blind sizing over- or
  under-bets.

### Phase 4 — Estimation error (the main event)
- Evaluate every sizing rule under several "truths": model q at x1 / x1.5 / x2, market mid, and
  blends w·q + (1 − w)·mid. Sizing uses one q, and outcomes are drawn from another.
- Metrics: expected log growth, mean and SD of PnL, Sharpe, P(loss), capital used.
- Robust alpha: shrink q toward the mid before sizing. Plot the growth-rate loss vs. the shrinkage
  weight under each truth. Connect to deck "Robust Alpha" and "Sharpe Ratio Efficiency".
- Fractional Kelly as the practitioner's version of the same idea.
- **Learn:** why practitioners trust Ω more than μ, and how much edge you need before sizing up is
  worth it.

### Phase 5 (optional) — Extensions
- Add the division contracts to the book (nested events, Cov = P(A)(1 − P(B))).
- A time series of the optimal book across daily snapshots: turnover, and how stable the weights are.
- A weekly game slate (needs a per-game probability export), where Ω is ~diagonal.

### Settlement check (after ~2027-01-11)
Score each rule's book on the realized outcomes. **This is one season — one correlated draw — and
it's an anecdote, not evidence.** Say so in the write-up.

---

## Conventions

- Python 3.12. numpy, scipy, pandas, matplotlib, pytest, and cvxpy (optional).
- Layout:
  ```
  src/mvo_binaries/   data.py  contracts.py  mvo.py  kelly.py  evaluate.py  plots.py
  scripts/            sync_data.py  run_phase*.py
  tests/
  docs/notes/  docs/writeups/
  data/raw/  data/processed/   (gitignored)
  ```
- Work in **contract units and dollars**. Do per-dollar returns only for the Phase 2 single-contract
  analysis.
- Research bankroll W₀ is a config value (default $1,000), separate from the real $20 prod cap.
- Prices: YES at ask, NO at 1 − bid (conservative, taker-like). The fee model is a config setting.
  Verify the current Kalshi fee schedule before using it, and don't hard-code it.
- Fix random seeds. Every plot is regenerated by a script, never hand-edited.
- Charts are simple and labeled with units. One idea per figure.

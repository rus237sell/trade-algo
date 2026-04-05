# trade-algo | Full Diagnostic Report
**Date:** April 2026
**Purpose:** Deep-research reference — systemic failure analysis and full-stack integration roadmap

---

## Executive Summary

The system has a coherent architecture on paper but breaks down at every major integration seam. The individual modules (pair finder, spread model, regime filter, Markov switcher, meta-labeler) each work in isolation. None of them are wired together into a unified pipeline that shares state, enforces economic constraints, or coordinates their outputs. The result is a backtest that generates 202 trades across 3 economically meaningless pairs, misses the entire ML filtering layer (no pair has enough labeled trades), and loses 0.58% per year on an unfiltered strategy — and loses 13.31% when the regime filter is naively multiplied against micro-sized positions that still bleed full transaction costs.

The problems are layered: wrong inputs feed into correct processes, producing outputs that are statistically valid but economically useless.

---

## Systemic Problem #1 — Pair Selection Is Selecting the Wrong Pairs

### What is happening

`research/pairs_finder.py` runs Engle-Granger cointegration across all 153 combinations of the 18-symbol universe over a fixed 252-day formation window starting at day 0 (roughly 2010). It accepts any pair that passes EG cointegration, ADF stationarity, and OU half-life < 120 days.

The three pairs that passed:
- **GS / WMT** — Goldman Sachs (investment bank) and Walmart (mass-market retail). Beta 17.39. No shared economic driver whatsoever.
- **JPM / MSFT** — JPMorgan (commercial bank) and Microsoft (enterprise software). Beta 0.84. Different industries, different cycles.
- **MS / MSFT** — Morgan Stanley and Microsoft. Beta 0.79. Shares the ticker abbreviation, not the business.

The economically sensible pairs in the universe — KO/PEP, XOM/CVX, GS/MS, HD/LOW, WMT/TGT, MCD/YUM — all **failed** because they have 14 years of structural breaks (2020 COVID shock, 2022 rate shock) that destroy cointegration when tested over the full 2010-2024 window.

### Why this is a problem

Statistical cointegration without economic cointegration is a coincidence. When the coincidence ends — and it always does — the pair diverges without reverting. GS/WMT beta 17.39 means every $1 long GS is hedged by $17.39 short WMT. The spread is almost entirely driven by whatever GS does unilaterally. Any persistent trend in GS becomes a directional bet with 17:1 leverage embedded in the hedge ratio.

### Evidence from the backtest

- Win rate: 32.7% on the baseline run. Mean-reversion strategies on valid pairs typically produce 55-65% win rates.
- Max drawdown: -84.15% on the baseline, -88.06% on the DFA run.
- Profit factor: 0.954 baseline (losses slightly exceed wins). Not the profile of a mean-reversion edge.
- DFA composite score for all three pairs: ~0.007 out of 1.0. The regime filter is correctly identifying these as trending assets — GS and WMT individually trend, and the spread between them trends.

### Root cause in the code

`pairs_finder.py` line 119: `for y_sym, x_sym in itertools.combinations(symbols, 2)` — no sector constraint, no economic grouping, full combinatorial scan. The formation window `prices.iloc[0:252]` is 2010 data. The backtest runs 2010-2024. Any cointegration relationship found in one early year of a 14-year window is likely to break.

`sector_pairs.py` exists and is more correct — it enforces sector membership, residualizes against the sector ETF, and applies K-means clustering within sectors. **It is never called by `run.py`.** It is orphaned code.

---

## Systemic Problem #2 — ML Meta-Labeler Is Permanently Bypassed

### What is happening

`run.py` phase 4 checks `if len(trade_labels_df) < config.ML_MIN_TRAIN_SAMPLES` and skips ML filtering for any pair below 80 labeled trades. All three active pairs have 64-69 trades total across the full 14-year backtest — they never reach the 80-trade minimum.

The `MetaLabeler` class is well-designed. It does proper walk-forward validation, tracks OOS AUC, and applies probability-scaled position sizing. It runs exactly zero times in this backtest.

### Why this is a problem

The ML layer is the system's primary risk management mechanism. Without it, every z-score signal that crosses ±2.0 is taken at full size, regardless of market context, VIX level, OU half-life deterioration, or spread momentum. The primary signal has a 32.7% win rate without ML filtering. That is not a trading strategy — that is a coin flip weighted toward losing.

### Root cause in the code

Two compounding problems:

1. **Too few trades per pair.** At ~5 trades/year per pair, clearing 80 labeled trades takes 16 years of a single pair. The minimum threshold `ML_MIN_TRAIN_SAMPLES = 80` in `config.py` is calibrated for a scenario where the system trades 8+ pairs simultaneously, generating 40+ trades/year pooled. The current pipeline runs pairs one at a time and never pools their trade history.

2. **No pooling logic.** `meta_labeler.py`'s `build_features()` takes features for one pair at a time. There is no code to stack labeled trades across pairs before training. The MetaLabeler is architecturally per-pair rather than cross-pair.

The fix described in the README — "Pool trade samples across all 8 instruments" — is conceptually correct but has no corresponding code in the pipeline.

---

## Systemic Problem #3 — Regime Filter and Markov Model Are Orphaned Modules

### What is happening

`signals/regime_filter.py` — complete DFA-1 + sigmoid scoring implementation. Never imported in `run.py`.

`signals/markov_regime.py` — 2-state Markov-switching model with proper filtered probabilities, 1-day lag, regime-conditional allocation. Never imported in `run.py` or `run_strategies.py`.

`run_dfa.py` is a standalone test script that applies the regime filter **after** signals are generated, multiplying raw signals by the DFA composite score. This multiplication approach is incorrect for the following reason:

When DFA correctly identifies a trending asset and returns a score of 0.007, the signal is not eliminated — it is scaled to 0.7% of its original size. At 0.007 × original signal, the position size is negligible, but the system still enters and exits the trade, paying full transaction costs (14 BPS round-trip) on a position too small to cover them. 202 trades × 14 BPS on near-zero positions = 202 negative-cost events with no offsetting profit. This is what turned -0.58% annual return into -13.31%.

### What the correct integration looks like

The regime filter should act as a **threshold gate with proportional sizing**, not a pure multiplier:

```
if composite_score < 0.15:
    signal = 0          # do not enter, save transaction costs
elif composite_score < 0.50:
    signal = signal * composite_score   # reduced size
else:
    signal = signal     # full size
```

The Markov model should control **capital allocation split** between the mean-reversion leg and a (currently missing) momentum leg, not individual trade entry decisions. The allocation logic in `markov_regime.py` is correct for this purpose but has no strategy to hand its outputs to.

### The missing momentum leg

The Markov model produces:
- Trending regime → 70% momentum / 30% mean-reversion
- Ranging regime → 30% momentum / 70% mean-reversion

`run_strategies.py` exists but the codebase has no functional momentum strategy. `strategies/rl_trader.py` is an RL-based pairs trader (not momentum). `strategies/dual_class_arb.py` is dual-class share arbitrage. There is no trend-following module. The allocation framework allocates to a momentum leg that does not exist — the 70% trending-regime allocation goes to zero.

---

## Systemic Problem #4 — Cointegration Window Is Static and Stale

### What is happening

`run.py` line 57: `accepted_pairs = find_cointegrated_pairs(prices, 0, formation_end)` where `formation_end = 252`. This is a fixed single-pass formation on the first 252 days of data, which is 2010. The resulting pairs are then traded through 2024 — 14 years of out-of-sample data on pairs identified in one early year.

There is no rolling re-evaluation of pair stability. A pair that was cointegrated in 2010 but broke down structurally in 2015 will keep generating signals through 2024 with no mechanism to detect or respond to the breakdown.

### Evidence from the code

`config.py` defines `FORMATION_PERIOD_DAYS = 252` and `ROLLING_BETA_WINDOW = 60`. The rolling OLS beta in `spread_model.py` does update the hedge ratio over time, which partially compensates — but it cannot fix a pair that is no longer cointegrated. A rolling beta on a non-cointegrated pair is just a rolling regression on noise.

### What is needed

Walk-forward pair selection: re-scan for cointegrated pairs every quarter on a rolling 252-day window. Retire pairs whose rolling ADF p-value drifts above 0.10. This is what prevents trading GS/WMT in 2022 on a relationship that may have been spurious since 2013.

---

## Systemic Problem #5 — No Factor Residualization Before Regime Testing

### What is happening

`rolling_dfa()` in `regime_filter.py` computes DFA on raw log-returns of individual stock prices. GS, WMT, JPM, MSFT all contain a large market factor (SPY beta), sector factor (XLF, XLK beta), and idiosyncratic component. DFA on raw returns measures the persistence of the total return, which is dominated by the market factor. The market itself trends — its Hurst exponent is consistently above 0.5.

This means DFA is correctly reporting that these stocks trend, but it is measuring market-driven trending rather than spread-specific trending. The correct target for DFA is the spread residual — the component that is supposed to mean-revert.

### What is needed

1. Remove market and sector factor from stock returns using a rolling 60-day OLS regression against SPY (and optionally the sector ETF).
2. Run DFA on the idiosyncratic residual.
3. An idiosyncratic residual with DFA < 0.5 is genuinely anti-persistent, not just a market-correlated stock in a ranging week.

`sector_pairs.py` already implements `residualize_against_sector()` which does this for the pair identification step. The same logic needs to be applied before feeding returns to `rolling_dfa()`.

---

## Systemic Problem #6 — Survivorship Bias in the Universe

### What is happening

`config.py` defines the universe as 18 symbols drawn from today's S&P 500 constituents. All 18 are currently large, liquid, surviving companies. Backtesting from 2010 on stocks that are known survivors in 2026 overstates pair stability and cointegration persistence — companies that failed, were acquired, or fell out of the index are excluded by construction.

This is acknowledged in the README: "Universe is drawn from today's stock list — survivorship bias overstates pair stability pre-2024."

### Why it matters for interpretation

The 32.7% win rate and -0.58% annual return are optimistic relative to what the strategy would have produced on a live universe. Every stat in the backtest should be read with a survivorship discount applied. The true performance on a forward-looking universe would be worse.

---

## Systemic Problem #7 — Transaction Cost Model Is Flat-BPS, Not Impact-Aware

### What is happening

`config.py`: `SLIPPAGE_BPS = 5`, `COMMISSION_BPS = 2`. Total round-trip: 14 BPS. This is a constant applied regardless of trade size, market conditions, or liquidity.

For a $1M portfolio trading pairs with a 17:1 hedge ratio (GS/WMT), the actual dollar notional on each trade can be large relative to the daily volume of either stock. A flat-BPS model assumes infinite liquidity at the mid-price minus a fixed spread. Real market impact scales with the square root of trade size relative to average daily volume.

The square-root model (Almgren-Chriss): `impact_bps = eta * sigma * sqrt(Q / ADV)` where Q is trade size, ADV is average daily volume, sigma is daily volatility, and eta is a market impact coefficient (typically 0.5-1.0).

At $2M+ portfolio size, flat-BPS slippage materially understates real costs. At the $1M initial capital in config, the error is smaller but still present for the larger-beta pairs.

---

## Full-Stack Integration Architecture

### What a correctly integrated pipeline looks like

The pipeline has six layers that need to execute in sequence, sharing state across layers:

```
LAYER 0: Universe Construction
  - Sector-constrained universe (SECTOR_UNIVERSE from sector_pairs.py)
  - Walk-forward re-scan quarterly
  - Residualize returns against sector ETFs before testing

LAYER 1: Regime Classification (NEW — runs daily on portfolio returns)
  - Markov-switching model fit on rolling 2-year equal-weighted portfolio return
  - Output: filtered marginal P(trending), P(ranging) — lagged 1 day
  - Regime label: trending / ranging / uncertain
  - Feeds into capital allocation split (Layer 5)

LAYER 2: Pair Identification (quarterly walk-forward)
  - sector_pairs.py: find_sector_pairs() per sector
  - Residualize against sector ETF before cointegration test
  - Rolling ADF health check: retire pairs where p > 0.10
  - Output: list of active pairs with beta, half-life, sector label

LAYER 3: Signal Generation (daily)
  - spread_model.py: rolling OLS beta, z-score
  - regime_filter.py: DFA on idiosyncratic residuals (not raw prices)
  - composite_score(): threshold gate (skip if < 0.15), proportional sizing above
  - Output: regime-filtered, scaled signals per pair

LAYER 4: Meta-Labeling (quarterly retrain)
  - Pool labeled trades across all active pairs (cross-pair features)
  - Walk-forward on pooled sample — minimum 80 trades total, not per pair
  - ML probability overlay on filtered signals
  - Output: probability-scaled final position sizes

LAYER 5: Capital Allocation (daily)
  - regime_allocation() from markov_regime.py feeds Markov regime weights
  - Mean-reversion allocation = total_capital × (MR weight from Markov)
  - Split MR allocation across active pairs by ML confidence (higher P = larger share)
  - Momentum allocation (currently zero — needs a trend-following module)

LAYER 6: Risk Controls (daily)
  - circuit_breaker.py: portfolio-level drawdown limit
  - position_sizing.py: Kelly criterion or fixed-fraction on individual pairs
  - Stop-loss: z-score > 3.5 closes position regardless of ML confidence
```

### Module connection map

| Module | Current State | Connected To | Should Connect To |
|--------|--------------|--------------|-------------------|
| `pairs_finder.py` | Active, wrong output | `run.py` L2 | `sector_pairs.py` as replacement |
| `sector_pairs.py` | Orphaned | Nothing | `run.py` L2 |
| `regime_filter.py` | Orphaned | `run_dfa.py` only | `run.py` L3 (with threshold gating) |
| `markov_regime.py` | Orphaned | Nothing | `run.py` L1 and L5 |
| `meta_labeler.py` | Active, never trains | `run.py` L4 | `run.py` L4 with pooled input |
| `kalman_filter.py` | Unclear usage | Unknown | L3 for Kalman-smoothed z-score |
| `circuit_breaker.py` | Exists | Unknown | L6 |
| `position_sizing.py` | Exists | Unknown | L6 |

---

## Implementation Roadmap (Ordered by Impact)

### Priority 1 — Fix Pair Selection (replaces bad inputs)

Replace `find_cointegrated_pairs()` call in `run.py` with `scan_all_sectors()` from `sector_pairs.py`. Add a rolling formation window that re-runs quarterly. Add ADF health check as a daily maintenance step that retires pairs above p=0.10.

This is the single highest-leverage fix. The entire downstream pipeline — regime filtering, meta-labeling, backtesting — is running on garbage inputs. Correct inputs will immediately change every downstream metric.

### Priority 2 — Fix Regime Filter Integration (replaces the multiplier bug)

Replace the `run_dfa.py` pure-multiplication approach with a threshold gate in the main pipeline:

1. Apply DFA to the spread series (not individual stock prices)
2. Use `dfa_percentile_threshold()` to get a boolean entry gate
3. When the gate is open, scale position by `composite_score` (proportional sizing)
4. When the gate is closed, skip the trade entirely — no position, no cost

This eliminates the 202-trade bleeder on near-zero positions.

### Priority 3 — Pool Labeled Trades for ML

Add cross-pair feature stacking in `run.py` phase 4. After building `trade_labels_df` for each pair, append to a master `all_labels` DataFrame with a pair_id column. Train MetaLabeler on the pooled set. Generate per-pair probability estimates using the pooled model.

At 5 trades/year × 8 pairs × 14 years = 560 labeled trades, the ML layer will have adequate history and can begin contributing after the first 80 pooled trades (~2 years).

### Priority 4 — Wire Markov Model into Allocation

In `run.py` after phase 3, add:

```python
from signals.markov_regime import rolling_regime_fit, regime_allocation

# fit on equal-weighted portfolio return
portfolio_returns = ...  # daily PnL across all pairs
regime_probs = rolling_regime_fit(portfolio_returns)
alloc_weights = regime_allocation(regime_probs)

# alloc_weights['mean_reversion_weight'] scales total MR capital
# alloc_weights['momentum_weight'] is reserved for a future momentum module
```

This connects the Markov model to actual allocation decisions without requiring the momentum leg to exist first. The momentum allocation simply sits at zero until a trend module is built.

### Priority 5 — Add Variance Ratio Test as Secondary Filter

`arch` is already in `requirements.txt`. The variance ratio test from `arch.unitroot.VarianceRatio` is a fast, robust secondary check on whether a spread is genuinely anti-persistent. Run it alongside DFA and require both to agree before passing a signal. This reduces false positives from DFA alone.

### Priority 6 — Add a Momentum Strategy

Without this, 60-80% of market time (trending regime) is fully unallocated. The Markov model pushes 70% of capital toward momentum in trending regimes, and that 70% earns nothing. A simple implementation: cross-sectional momentum on the sector ETFs (XLE, XLF, XLK, XLP, XLY) ranked monthly — long top 2, short bottom 2. This is a well-documented, positive-expected-value strategy that has negative correlation with mean-reversion in normal market conditions.

### Priority 7 — Rolling Formation Window

Quarterly re-scan for cointegrated pairs. Retire broken pairs. Add new ones as market structure evolves. Prevents trading 14-year-old cointegration relationships that may have broken in 2015.

---

## Quantified Evidence from Current Backtests

| Metric | Baseline (run.py) | DFA Run (run_dfa.py) | Target Range |
|--------|-------------------|----------------------|--------------|
| Annualized Return | -0.58% | -13.31% | +8% to +18% |
| Sharpe Ratio | -0.016 | -5.274 | +0.8 to +1.5 |
| Max Drawdown | -84.15% | -88.06% | -15% to -30% |
| Win Rate | 32.7% | 2.0% | 55% to 65% |
| Profit Factor | 0.954 | 0.059 | 1.5 to 2.5 |
| Total Trades | 202 | 202 | 40+ per year |
| Beta vs Market | -0.485 | -0.003 | < ±0.15 |
| ML Trades | 0 | 0 | 100% of trades |
| DFA Score (avg) | N/A | 0.007 | >0.40 target |
| Pairs Used | 3 (all spurious) | 3 (all spurious) | 8-12 (sector-constrained) |

The DFA run is worse than the baseline not because DFA is wrong but because the threshold gating is missing and the pairs themselves are wrong. DFA at 0.007 is correctly saying "these are trending assets — do not trade them." The pipeline ignores that advice and trades them at 0.7% size, paying full costs.

---

## What a Realistic Outcome Looks Like After Full Integration

Based on the literature this system is attempting to implement:

- Sector-constrained pairs trading on 8-12 pairs: Sharpe 0.8-1.3 annualized (documented in multiple academic papers at 40-50 trades/year per pair)
- Meta-labeling overlay (Lopez de Prado): additional 15-25% Sharpe improvement on top of primary signals
- Regime-adaptive allocation (Markov): additional 30-40% Sharpe improvement by allocating away from mean-reversion during trending periods and toward momentum
- Combined target: Sharpe 1.2-1.8 after transaction costs, max drawdown -20% to -35%, 200+ annual trades across all pairs

These are research-grade estimates on clean inputs. Survivorship bias means real forward performance will be lower. The slippage model needs to be upgraded at capital above $2M.

---

## Files That Need to Change

| File | Change Type | Priority |
|------|------------|----------|
| `run.py` | Major rewrite — add all six layers, wire all modules | 1 |
| `research/pairs_finder.py` | Add rolling formation window and ADF health check | 2 |
| `signals/regime_filter.py` | Apply to spread residuals, not raw prices; add threshold gate | 3 |
| `run.py` phase 4 | Pool labeled trades cross-pair before ML training | 4 |
| `signals/markov_regime.py` | Wire into allocation, not just standalone | 5 |
| `config.py` | Add SECTOR_SCAN flag, ROLLING_FORMATION_WINDOW, DFA_THRESHOLD_GATE | 6 |
| `run_dfa.py` | Replace with correct threshold gating logic | 7 |
| `strategies/` | Add momentum strategy module (sector ETF cross-sectional) | 8 |

---

*This report is a snapshot of the repository state as of April 2026 after the DFA-1 regime filter integration. All performance figures are from vectorized backtests on yfinance data 2010-2024 with 14 BPS round-trip costs and $1M initial capital.*

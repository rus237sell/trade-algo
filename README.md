# trade-algo

A systematic equity research pipeline built around pairs trading — find two structurally related stocks (KO/PEP, GS/MS, XOM/CVX), measure when their spread stretches beyond normal, and bet on reversion. A machine learning layer filters signals by estimating the probability a given trade will actually work before sizing into it.

---

## What It Does

Layer 1 identifies cointegrated pairs and generates z-score signals when spreads diverge. Layer 2 is a Random Forest meta-labeler that filters those signals using market context — volatility levels, VIX, z-score momentum, OU half-life. Position size scales with predicted success probability rather than being binary on/off.

---

## What We Started With

The first version was a naive SMA trader — buy when price drops below the 5-period moving average, sell when it rises above. It worked occasionally in sideways markets and got wrecked in trending ones. No regime awareness, fixed spreads, position limits of 20, no memory of market state.

That exposed the core problem: a mean-reversion strategy has no edge in a trending market. It will just keep fading a move that keeps going. Everything after that was solving this problem progressively.

---

## Flaws Found

**Flaw 1 — The Hurst estimator was lying (3 flaws in 1)**

The system used classical R/S analysis to compute the Hurst exponent and gated trades behind H < 0.45. Problem: R/S analysis inflates H by +0.05 to +0.10 on any stock with upward drift. Since large-cap US equities generally trend upward over time, the H < 0.45 gate was essentially unreachable — it was blocking 77–87% of trading days. The system thought it was being disciplined; it was just not trading.

**Flaw 2 — Binary AND gating killed volume**

Stacking hard filters multiplicatively is self-defeating. If H < 0.45 passes 30% of days, and ADX < 25 passes 40% of days, the combined AND gate passes 12%. The observed result was approximately 0.6 trades per year — statistically meaningless. You need a minimum of 30 trades per year per instrument to distinguish signal from noise. At 0.6/year, reaching 30 independent trades takes 50 years of data.

**Flaw 3 — Trending regimes were completely discarded**

Stocks spend roughly 60–80% of time in trending conditions (H > 0.5). The original design filters all of that out and only trades the ranging minority. A mean-reversion strategy running alone is leaving most of the market's time on the table. The diversification math on combining momentum and mean-reversion is well-documented: at a correlation of –0.3 between the two legs, a combined portfolio yields roughly 69% Sharpe improvement over either alone.

---

## What Was Changed

**`signals/regime_filter.py`** — R/S Hurst replaced with DFA-1 (Detrended Fluctuation Analysis via `nolds`). DFA explicitly fits and removes a linear trend within each non-overlapping box before measuring fluctuation scaling, making it drift-robust by construction. Bias on N=1000 is +0.02 to +0.04 versus R/S at +0.05 to +0.10. The fixed H < 0.45 threshold is replaced with a percentile gate: enter mean-reversion when the current DFA exponent falls in the bottom 30th percentile of its own trailing 2-year distribution. This adapts automatically across different assets and market regimes. Binary AND gate replaced with continuous sigmoid weights — the composite score in [0,1] scales position size proportionally rather than gating it off entirely.

**`signals/markov_regime.py`** — New file. 2-state Markov-switching model (Hamilton 1989, implemented via statsmodels) classifies each day as trending (low-vol, positive drift) or ranging (high-vol, choppy). Only filtered marginal probabilities are used — never smoothed probabilities, which incorporate future data. A one-day lag is applied before acting on any estimates. Allocation shifts 70/30 toward whichever regime the model identifies, probability-weighted to avoid cliff-edge allocation changes from noisy estimates near the decision boundary.

---

## Backtesting Notes

Results are research-grade, not production-grade. Key constraints:

- Universe is drawn from today's stock list — survivorship bias overstates pair stability pre-2024
- Cointegration relationships break during structural shocks (2020, 2022); rolling beta and half-life filter partially mitigate this
- ML meta-labeler requires 80+ labeled trades before training; pairs that trade infrequently run on unfiltered primary signals for extended periods
- Flat-BPS slippage model understates real market impact at larger capital allocations

---

## Improvements

- Factor-residualize returns before computing DFA — remove Fama-French common factor exposure first, then run DFA on the idiosyncratic residual. This is the cleanest way to isolate genuine mean-reversion signal from market-wide drift
- Add variance ratio test (via `arch`) as a secondary validation pass on the DFA signal
- Pool trade samples across all instruments — at 5 trades/year/stock, 8 instruments yields 40 annual trades, clearing the minimum threshold for CPCV validation
- Reduce CPCV to N=4 folds with embargo = max(holding period, DFA lookback window)
- Replace flat-BPS slippage with a square-root market impact model for capital allocations above ~$2M

---

## Integration Results

What was broken and what was fixed across tasks 2–10.

**Before (baseline run.py):**

| Metric | Value |
|---|---|
| Annualized Return | -0.58% |
| Sharpe Ratio | -0.016 |
| Max Drawdown | -84.15% |
| Win Rate | 32.7% |
| Profit Factor | 0.954 |
| Total Trades | 202 |
| ML Trades Filtered | 0 (never trained) |
| Pairs | GS/WMT, JPM/MSFT, MS/MSFT (all spurious) |

**After (all fixes applied):**

Target ranges based on sector-constrained literature with meta-labeling overlay:

| Metric | Baseline | Target |
|---|---|---|
| Annualized Return | -0.58% | +8% to +18% |
| Sharpe Ratio | -0.016 | +0.8 to +1.5 |
| Max Drawdown | -84.15% | -15% to -30% |
| Win Rate | 32.7% | 55% to 65% |
| Profit Factor | 0.954 | 1.5 to 2.5 |
| Total Trades | 202 | 200+ across all pairs |

**Fix 1 — pair selection (task 2):** switched from naive combinatorial scan to sector-constrained search. Economically related pairs only: KO/PEP, GS/MS, XOM/CVX, HD/LOW. Beta bounded 0.3–3.0, half-life bounded 3–42 days. Eliminates GS/WMT (beta 17.39) and other coincidental pairs.

**Fix 2 — rolling pair health (task 3):** quarterly rescan with rolling ADF health check. Pairs whose ADF p-value drifts above 0.10 are retired. Prevents trading 14-year-old relationships that may have broken structurally in 2015–2018.

**Fix 3 — regime filter gating (task 4):** replaced raw DFA score multiplication with threshold gate. Below 0.15: no trade (cost saved). Between 0.15–0.50: proportional sizing. Above 0.50: full signal. Eliminated the -13.31% bug where near-zero positions paid full transaction costs on every entry/exit.

**Fix 4 — ML pooling (task 5):** pooled labeled trades across all pairs before ML training. At 5 trades/year × 8 pairs × 14 years = 560 pooled trades, the 80-trade minimum is cleared within the first 2 years. ML meta-labeling now runs on every trade.

**Fix 5 — Markov allocation (task 6):** Markov-switching model classifies each day as trending or ranging. Mean-reversion signals are scaled by the Markov MR weight (0.70 in ranging, 0.30 in trending). Trending capital is redirected to momentum (task 8) rather than sitting idle.

**Fix 6 — variance ratio (task 7):** Lo-MacKinlay VR test at lags 2, 5, 10, 20 acts as a secondary check on DFA. If fewer than 2 of 4 lags confirm anti-persistence, the DFA composite score is halved before the threshold gate. Reduces false positives on short spread histories.

**Fix 7 — momentum strategy (task 8):** cross-sectional TSMOM on sector ETFs (XLP, XLF, XLE, XLK, XLY). 12-1 month lookback, long top 2 ETFs, short bottom 2, volatility-targeted, monthly rebalance. Captures trending regimes that mean-reversion misses.

**Fix 8 — factor residualization (task 9):** DFA now runs on Fama-French residualized spread returns, not raw prices. Removes market-wide drift from the DFA computation so it measures spread-specific persistence rather than market factor persistence.

**Fix 9 — slippage model (task 10):** Almgren-Chriss square-root impact model replaces flat 14 BPS. Impact scales with sqrt(participation rate). More realistic at any capital size above $500K.

Survivorship bias (all 18 symbols are current S&P 500 survivors) means all performance figures are optimistic by some margin. Forward results on a live universe will be lower.

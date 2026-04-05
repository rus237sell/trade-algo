# quant-research

An offline quantitative research pipeline combining classical econometrics with machine learning for systematic equity trading. The architecture separates signal generation from signal filtering — a two-model design that mirrors the research process at systematic funds.

---

## Research Objective

The central question this project addresses: when a statistically valid mean-reversion signal fires, can a secondary machine learning model determine whether market conditions at that moment are likely to produce a profitable outcome?

The answer to that question is operationalized through a dual-layer architecture: a cointegration-based primary model that generates candidate trades, and a Random Forest meta-labeler that filters and sizes those trades based on contextual market features.

---

## Architecture Overview

```
raw prices
    |
    v
[Cointegration Layer]
  ADF test + Engle-Granger
  Rolling OLS hedge ratio
  Z-score signal generation
    |
    v
primary signal: {-1, 0, +1}
    |
    v
[Meta-Labeling Layer]
  Feature extraction at signal time
  Walk-forward Random Forest classifier
  P(success) estimation
    |
    v
filtered signal: scaled by P(success), zeroed if P < threshold
    |
    v
[Backtesting Engine]
  Slippage + commission simulation
  Trade-level P&L extraction
  Equity curve
    |
    v
performance metrics
```

---

## Layer 1: Cointegration

Two assets Y and X are cointegrated if their long-run relationship is stationary, meaning:

```
Y_t = alpha + beta * X_t + epsilon_t
```

where epsilon_t ~ I(0) (stationary). The spread S_t = Y_t - beta * X_t will revert to its mean after deviations, providing a tradeable signal.

**Pair Selection** applies three filters sequentially:

1. Engle-Granger cointegration test (p < 0.05)
2. Augmented Dickey-Fuller test on the residual spread (robustness check)
3. Ornstein-Uhlenbeck half-life filter (half_life < 120 trading days)

The OU half-life is estimated from the discrete-time regression:

```
delta_S_t = a + b * S_{t-1} + epsilon_t
half_life = -ln(2) / b
```

Pairs with slow mean reversion (long half-life) are discarded because they tie up capital for extended periods with poor risk-adjusted returns.

**Time-varying beta** is estimated via rolling 60-day OLS rather than a static formation-period estimate. This allows the hedge ratio to adapt to structural shifts in the relationship between the two assets, reducing the directional leakage that a stale fixed beta would introduce.

**Signal generation** uses rolling z-score thresholds:

- Long spread (+1): z < -2.0 (spread below equilibrium)
- Short spread (-1): z > +2.0 (spread above equilibrium)
- Exit: |z| < 0.5
- Stop-loss: |z| > 3.5 (spread diverging — structural break possible)

---

## Layer 2: Meta-Labeling

Meta-labeling (Lopez de Prado, 2018) treats the primary model's signal as a candidate rather than an execution instruction. A secondary model then estimates the probability that this specific candidate trade will be profitable given the current market context.

**Label Construction** simulates each primary trade to completion and assigns:

- Label 1: trade closed profitably (spread reverted in the direction of the bet)
- Label 0: trade closed at a loss (spread diverged or hit stop-loss)

**Feature Set** at trade entry:

| Feature | Description |
|---|---|
| z_at_entry | Z-score magnitude at signal time |
| spread_volatility | 20-day rolling std of spread returns |
| z_velocity | First difference of z-score (rate of divergence) |
| z_acceleration | Second difference of z-score (is divergence accelerating?) |
| vix_level | CBOE VIX at entry date |
| vix_percentile | 252-day rolling percentile rank of VIX |
| ou_half_life | 60-day rolling OU half-life estimate |

**Walk-Forward Validation** ensures no lookahead contamination. At each retraining step T, the model trains only on labeled trades in [0, T) and generates predictions for [T, T + step). This produces a fully out-of-sample probability series across the entire backtest period.

**Position Sizing** scales the primary signal by the predicted probability:

```
if P(success) < threshold:      position = 0
if P(success) >= threshold:     position = signal * (P - threshold) / (1 - threshold)
```

A trade with P = 0.85 at threshold = 0.60 receives 100% of the target allocation. A trade with P = 0.62 receives 8%. This makes high-confidence trades substantially larger than borderline ones.

---

## Backtesting Methodology

The engine is fully vectorized using pandas and numpy. All costs are applied on signal changes (entries and exits), not on a per-day basis.

**Transaction cost model:**

```
round_trip_cost = 2 * (slippage_bps + commission_bps) / 10,000
```

Default: 5 bps slippage + 2 bps commission per side = 14 bps round-trip.

**P&L calculation** per pair per day:

```
pnl_t = signal_{t-1} * (log_return_Y_t - beta_{t-1} * log_return_X_t)
```

The lagged signal and lagged beta prevent any execution-day lookahead.

---

## Performance Metrics

| Metric | Definition |
|---|---|
| Annualized Sharpe | (mean daily return / std daily return) * sqrt(252) |
| Annualized Sortino | (mean daily return / downside std) * sqrt(252) |
| Max Drawdown | Max peak-to-trough decline of equity curve |
| Calmar Ratio | Annualized return / |Max Drawdown| |
| Win Rate | Fraction of trades with positive net P&L |
| Profit Factor | Sum of winning trades / |sum of losing trades| |
| Beta | Cov(strategy, SPY) / Var(SPY) |

---

## Research Notes

This is a research prototype, not a production system. Several limitations apply:

**Survivorship bias.** The universe is drawn from today's stock list. Any company that was delisted or went bankrupt between 2010 and 2024 is absent, which overstates historical pair stability. A production system would require a point-in-time universe database.

**Regime dependence.** Cointegration relationships are not permanent. The 2020 COVID shock and 2022 rate cycle disrupted many structural relationships that appeared stable over the preceding decade. The rolling beta and half-life filter partially mitigate this, but structural breaks can invalidate a pair rapidly.

**ML sample size.** The meta-labeling model requires a minimum of 80 labeled trades before training begins. For pairs that trade infrequently, this means a substantial portion of the backtest period runs on unfiltered primary signals.

**Execution model.** The flat-BPS slippage model is a simplification. Actual market impact scales with trade size and liquidity conditions. For larger capital allocations, a more sophisticated market impact model (e.g., square-root law) would be necessary.

# all strategy and research parameters in one place

# universe — structurally related pairs for cointegration testing
UNIVERSE = [
    "KO",   "PEP",   # beverages
    "GS",   "MS",    # bulge-bracket banks
    "XOM",  "CVX",   # integrated oil majors
    "WMT",  "TGT",   # mass-market retail
    "JPM",  "BAC",   # commercial banking
    "HD",   "LOW",   # home improvement
    "MCD",  "YUM",   # QSR / fast food
    "MSFT", "ORCL",  # enterprise software
    "NKE",  "UAA",   # athletic apparel
]

# historical data window
START_DATE = "2010-01-01"
END_DATE   = "2024-06-01"

# cointegration testing
COINTEGRATION_PVALUE  = 0.05   # ADF significance level to accept a pair
FORMATION_PERIOD_DAYS = 252    # in-sample window used to test for cointegration

# spread modeling
ROLLING_BETA_WINDOW = 60       # OLS window for time-varying beta estimation
ZSCORE_WINDOW       = 60       # rolling window for z-score normalization
ZSCORE_ENTRY        = 2.0      # z-score threshold to enter a trade
ZSCORE_EXIT         = 0.5      # z-score magnitude to close a winning trade
ZSCORE_STOP         = 3.5      # stop-loss: spread diverging beyond this exits

# Ornstein-Uhlenbeck half-life filter
# pairs whose mean-reversion half-life exceeds this are discarded
MAX_HALF_LIFE_DAYS  = 120

# execution model
SLIPPAGE_BPS    = 5   # one-way slippage in basis points
COMMISSION_BPS  = 2   # one-way commission in basis points
INITIAL_CAPITAL = 1_000_000

# meta-labeling (ML overlay)
ML_PROB_THRESHOLD    = 0.60   # minimum predicted win probability to take the trade
ML_MIN_TRAIN_SAMPLES = 80     # minimum labeled trades before training the model
WALK_FORWARD_STEP    = 63     # retrain every quarter (trading days)
RF_N_ESTIMATORS      = 200
RF_MAX_DEPTH         = 4      # intentionally shallow to reduce overfit
RF_MIN_SAMPLES_LEAF  = 5

# DFA-1 regime filter (replaces R/S Hurst)
# R/S Hurst inflates exponents by +0.05 to +0.10 on trending equities —
# DFA-1 removes trends within each box and is drift-robust by construction.
DFA_WINDOW             = 504     # 2-year rolling window; bias +0.02 to +0.04
DFA_STEP               = 5      # recompute every 5 days
DFA_PERCENTILE_THRESH  = 0.30   # enter MR when DFA < 30th pct of own 2-year history

# Continuous sigmoid scoring (replaces binary AND gate)
# Binary gating at 40% pass rate each: 0.4^2 = 16% pass.
# Sigmoid weighting scales position continuously instead of hard on/off.
DFA_SIGMOID_THRESHOLD  = 0.50   # DFA ~ 0.50 is the random walk boundary
DFA_SIGMOID_SHARPNESS  = 20.0
ADX_SIGMOID_THRESHOLD  = 25.0
ADX_SIGMOID_SHARPNESS  = 0.3

# Markov-switching regime model
# Stocks spend ~60-80% of time trending. This captures those periods
# instead of discarding them. Allocation shifts toward dominant regime.
MARKOV_ESTIMATION_DAYS = 504    # rolling estimation window (2 years)
MARKOV_REFIT_DAYS      = 21     # refit monthly
MARKOV_PROB_THRESHOLD  = 0.70   # probability to act on regime shift (reduce whipsaw)
MOM_WEIGHT_TRENDING    = 0.70   # momentum allocation in trending regime
MR_WEIGHT_TRENDING     = 0.30   # mean-reversion allocation in trending regime

# sector pair selection (task 2)
USE_SECTOR_PAIRS             = True
SECTOR_RESIDUALIZE           = True
SECTOR_RESIDUALIZE_WINDOW    = 60
BETA_MIN                     = 0.3
BETA_MAX                     = 3.0
HALFLIFE_MIN                 = 3
HALFLIFE_MAX                 = 42

# rolling pair health (task 3)
FORMATION_WINDOW             = 252
RESCAN_INTERVAL              = 63
ADF_RETIRE_THRESHOLD         = 0.10
ADF_HEALTH_WINDOW            = 126

# regime filter gating (task 4)
DFA_MIN_THRESHOLD            = 0.15
DFA_FULL_THRESHOLD           = 0.50
DFA_PERCENTILE               = 0.30

# ML pooling (task 5)
ML_POOL_ACROSS_PAIRS         = True
ML_RETRAIN_INTERVAL          = 63

# Markov allocation (task 6)
MARKOV_WINDOW                = 504
MARKOV_REFIT_INTERVAL        = 21
MARKOV_SWITCH_THRESHOLD      = 0.70
MARKOV_TRENDING_MR_WEIGHT    = 0.30
MARKOV_TRENDING_MOM_WEIGHT   = 0.70
MARKOV_RANGING_MR_WEIGHT     = 0.70
MARKOV_RANGING_MOM_WEIGHT    = 0.30

# momentum strategy (task 8)
MOMENTUM_LOOKBACK            = 252
MOMENTUM_SKIP                = 21
MOMENTUM_LONG_N              = 2
MOMENTUM_SHORT_N             = 2
MOMENTUM_TARGET_VOL          = 0.10
MOMENTUM_REBALANCE_INTERVAL  = 21
MOMENTUM_ETFS                = ['XLP', 'XLF', 'XLE', 'XLK', 'XLY']

# almgren-chriss slippage (task 10)
USE_SQRT_IMPACT              = True
IMPACT_ETA                   = 0.75
MAX_PARTICIPATION_RATE       = 0.05
MIN_SLIPPAGE_BPS             = 2.0

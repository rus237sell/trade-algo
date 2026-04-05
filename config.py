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
ML_PROB_THRESHOLD   = 0.60   # minimum predicted win probability to take the trade
ML_MIN_TRAIN_SAMPLES = 80    # minimum labeled trades before training the model
WALK_FORWARD_STEP   = 63     # retrain every quarter (trading days)
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 4      # intentionally shallow to reduce overfit
RF_MIN_SAMPLES_LEAF = 5

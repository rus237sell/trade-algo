"""
Kalman Filter Mean Estimator

Replaces the fixed SMA-20 with an adaptive filter that:
  1. Produces a theoretically grounded mean estimate with uncertainty bounds
  2. Updates in O(1) per timestep — compatible with live trading
  3. Requires no fixed lookback window — the filter self-tunes via the
     state covariance matrix

The Kalman z-score (price deviation / filter uncertainty) is strictly
superior to (price - SMA) / rolling_std because the denominator is not
an arbitrary rolling window — it is the filter's own estimate of how
uncertain it is about the current price level.

State model (local level + trend):
  x_t = [level, velocity]     (2x1)
  F   = [[1, 1], [0, 1]]      transition: level += velocity each step
  H   = [1, 0]                observation: we only see the level
  Q   = transition noise      (how fast can the true mean shift?)
  R   = observation noise     (how noisy are price observations?)

The ratio Q/R is the single tuning parameter:
  Q/R = 0.001  =>  SMA-50-like smoothness (slow to adapt)
  Q/R = 0.010  =>  SMA-20-like smoothness
  Q/R = 0.100  =>  SMA-10-like responsiveness (fast to adapt)

For pairs trading spreads (already mean-reverting), Q/R = 0.005-0.01
is the typical sweet spot. For single-asset mean reversion, Q/R = 0.01.
"""

import numpy as np
import pandas as pd


def run_kalman_filter(
    prices:       pd.Series,
    transition_cov: float = 0.01,    # Q — how fast can the true mean drift?
    observation_cov: float = 1.0,    # R — observation noise level
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Univariate Kalman filter with local level + trend state model.

    Returns:
      filtered_mean:  E[level | observations up to t]
      filtered_std:   sqrt(P[0,0]) — uncertainty in the level estimate
      kalman_zscore:  (price - filtered_mean) / filtered_std
                      this is the signal used for entry/exit decisions
    """
    n      = len(prices)
    vals   = prices.values.astype(float)

    # state covariance matrix (2x2)
    # P[0,0] = uncertainty in level
    # P[1,1] = uncertainty in velocity
    P = np.eye(2) * 1.0

    # state vector: [level, velocity]
    x = np.array([vals[0], 0.0])

    # transition and observation matrices
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    # noise matrices
    Q = np.array([[transition_cov, 0.0],
                  [0.0,            transition_cov * 0.1]])
    R = np.array([[observation_cov]])

    means  = np.empty(n)
    stds   = np.empty(n)

    for t in range(n):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        # update
        y = vals[t] - (H @ x)[0]                  # innovation
        S = (H @ P @ H.T + R)[0, 0]               # innovation covariance
        K = (P @ H.T / S).flatten()                # Kalman gain (2x1)

        x = x + K * y
        P = (np.eye(2) - np.outer(K, H)) @ P

        means[t] = x[0]
        stds[t]  = np.sqrt(max(P[0, 0], 1e-8))

    filtered_mean  = pd.Series(means, index=prices.index)
    filtered_std   = pd.Series(stds,  index=prices.index)
    kalman_zscore  = (prices - filtered_mean) / filtered_std

    return filtered_mean, filtered_std, kalman_zscore


def kalman_spread_model(
    y:              pd.Series,
    x:              pd.Series,
    transition_cov: float = 0.005,
    observation_cov: float = 1.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Kalman filter for pairs trading — estimates the time-varying beta
    (hedge ratio) between y and x simultaneously with the spread level.

    This is the Pairs Trading Kalman Filter from Elliot, Van Der Hoek
    and Malcolm (2005) and popularized by E.P. Chan.

    State: [beta, alpha] where spread = y - (beta * x + alpha)

    The beta updates as new price data arrives — no fixed rolling window.
    When the pair relationship shifts (e.g., one company issues equity),
    the filter adapts within a few days rather than waiting for the OLS
    window to roll.

    Returns:
      beta:    time-varying hedge ratio
      spread:  y - beta*x - alpha (the residual)
      mean:    filtered mean of the spread (should hover near 0)
      zscore:  spread deviation normalized by filter uncertainty
    """
    n    = len(y)
    yv   = y.values.astype(float)
    xv   = x.values.astype(float)

    # state: [beta, alpha]
    theta = np.array([1.0, 0.0])
    P     = np.eye(2) * 1.0
    Q     = np.eye(2) * transition_cov
    R     = observation_cov

    betas   = np.empty(n)
    alphas  = np.empty(n)
    spreads = np.empty(n)
    stds    = np.empty(n)

    for t in range(n):
        # observation vector: how does the state map to y?
        H = np.array([xv[t], 1.0])   # y_t = beta * x_t + alpha + noise

        # predict (random walk on state)
        P = P + Q

        # innovation
        y_hat = H @ theta
        innov = yv[t] - y_hat
        S     = H @ P @ H.T + R
        K     = P @ H / S

        # update
        theta = theta + K * innov
        P     = (np.eye(2) - np.outer(K, H)) @ P

        betas[t]   = theta[0]
        alphas[t]  = theta[1]
        spreads[t] = yv[t] - theta[0] * xv[t] - theta[1]
        stds[t]    = np.sqrt(max(S, 1e-8))

    idx     = y.index
    beta_s  = pd.Series(betas,   index=idx)
    spread  = pd.Series(spreads, index=idx)
    mean_s  = pd.Series(alphas,  index=idx)   # alpha is the spread mean
    std_s   = pd.Series(stds,    index=idx)
    zscore  = spread / std_s

    return beta_s, spread, mean_s, zscore

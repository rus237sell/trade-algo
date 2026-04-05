"""
Tiered Drawdown Circuit Breakers

Prevents catastrophic losses by reducing position sizes and halting
trading as drawdowns deepen. This is the institutional standard protocol
used by systematic trading desks at hedge funds.

The four tiers:
  -5%  drawdown: reduce position sizes by 25% (caution)
  -10% drawdown: reduce position sizes by 50% (defensive)
  -15% drawdown: halt ALL new entries (mandatory review)
  -20% drawdown: close all positions, move to cash (emergency stop)

After a circuit breaker triggers:
  - Resume at 50% position size when equity recovers to -12% from peak
  - Require 3-5 consecutive profitable days before scaling back to full size
  - This "gradual re-entry" prevents whipsaw after a false recovery

Daily loss limit: -2% of equity is a hard stop for that trading day.
If the portfolio drops 2% intraday, no new positions are opened for the
remainder of that session regardless of signals.

Research basis:
  - Standard protocol from International Trading Institute
  - Lopez de Prado 2018 — "Advances in Financial Machine Learning"
  - Institutional risk desk practice at multi-strategy hedge funds
"""

import numpy as np
import pandas as pd
from enum import IntEnum


class RiskTier(IntEnum):
    FULL       = 0    # normal — full position sizing
    CAUTION    = 1    # -5%  drawdown — 75% of normal size
    DEFENSIVE  = 2    # -10% drawdown — 50% of normal size
    HALT       = 3    # -15% drawdown — no new entries
    EMERGENCY  = 4    # -20% drawdown — close everything


TIER_THRESHOLDS = {
    RiskTier.CAUTION:   -0.05,
    RiskTier.DEFENSIVE: -0.10,
    RiskTier.HALT:      -0.15,
    RiskTier.EMERGENCY: -0.20,
}

TIER_SIZE_MULT = {
    RiskTier.FULL:      1.00,
    RiskTier.CAUTION:   0.75,
    RiskTier.DEFENSIVE: 0.50,
    RiskTier.HALT:      0.00,    # no new entries
    RiskTier.EMERGENCY: 0.00,    # emergency exit
}

DAILY_LOSS_LIMIT  = -0.02    # -2% daily hard stop
RECOVERY_THRESH   = -0.12    # resume partial trading when drawdown < 12%
RECOVERY_DAYS_REQ = 3        # consecutive profitable days before full-size


class CircuitBreaker:
    """
    Stateful circuit breaker that tracks drawdown and adjusts sizing.

    Usage in backtest loop:
      cb = CircuitBreaker(initial_equity=1_000_000)
      for date in dates:
          cb.update(current_equity)
          if cb.allows_new_entry():
              size_mult = cb.size_multiplier()
              # apply size_mult to position size before submitting order
    """

    def __init__(self, initial_equity: float):
        self.peak_equity        = initial_equity
        self.current_equity     = initial_equity
        self.session_start_eq   = initial_equity
        self.tier               = RiskTier.FULL
        self.consecutive_wins   = 0
        self.recovery_mode      = False
        self.daily_loss_hit     = False
        self._tier_history      = []
        self._drawdown_history  = []

    def update(self, equity: float, date=None) -> None:
        """Called once per bar with the current portfolio equity."""
        self.current_equity = equity
        self.peak_equity    = max(self.peak_equity, equity)

        drawdown = (equity - self.peak_equity) / self.peak_equity
        self._drawdown_history.append(drawdown)

        # determine tier from drawdown depth
        new_tier = RiskTier.FULL
        for tier in [RiskTier.EMERGENCY, RiskTier.HALT,
                     RiskTier.DEFENSIVE, RiskTier.CAUTION]:
            if drawdown <= TIER_THRESHOLDS[tier]:
                new_tier = tier
                break

        # recovery logic: once below HALT, require positive days before resuming
        if self.tier >= RiskTier.HALT and new_tier < RiskTier.HALT:
            if drawdown > RECOVERY_THRESH:
                self.recovery_mode = True

        if self.recovery_mode:
            # stay in DEFENSIVE until consecutive_wins requirement is met
            new_tier = max(new_tier, RiskTier.DEFENSIVE)
            if self.consecutive_wins >= RECOVERY_DAYS_REQ:
                self.recovery_mode  = False
                self.consecutive_wins = 0

        self.tier = new_tier
        self._tier_history.append(int(self.tier))

    def record_daily_pnl(self, daily_return: float) -> None:
        """
        Called with the day's P&L fraction to track the daily loss limit
        and consecutive win count for recovery mode.
        """
        self.daily_loss_hit = daily_return < DAILY_LOSS_LIMIT

        if daily_return > 0:
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0

    def reset_daily(self, new_session_equity: float) -> None:
        """Call at the start of each trading day to reset intraday limits."""
        self.session_start_eq = new_session_equity
        self.daily_loss_hit   = False

    def allows_new_entry(self) -> bool:
        """Returns True if new positions can be opened."""
        if self.daily_loss_hit:
            return False
        return self.tier < RiskTier.HALT

    def requires_emergency_exit(self) -> bool:
        """Returns True when all positions should be closed immediately."""
        return self.tier == RiskTier.EMERGENCY

    def size_multiplier(self) -> float:
        """
        Returns the fraction of normal position size to use.
        Apply this to the output of position_sizing.combined_position_weight().
        """
        return TIER_SIZE_MULT[self.tier]

    def current_drawdown(self) -> float:
        """Current drawdown from peak as a fraction."""
        if self.peak_equity == 0:
            return 0.0
        return (self.current_equity - self.peak_equity) / self.peak_equity

    def summary(self) -> dict:
        """Returns circuit breaker state summary for logging."""
        return {
            "tier":             self.tier.name,
            "drawdown":         self.current_drawdown(),
            "size_mult":        self.size_multiplier(),
            "allows_entry":     self.allows_new_entry(),
            "recovery_mode":    self.recovery_mode,
            "consecutive_wins": self.consecutive_wins,
        }


# ---- Vectorized version for backtesting -------------------------------------

def apply_circuit_breaker(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Vectorized circuit breaker for backtesting — applies tier logic
    across a full equity curve without a Python loop.

    Returns:
      size_multiplier: Series of position size fractions (0.0 to 1.0)
      tier_series:     Series of RiskTier values for analysis
    """
    peak       = equity_curve.cummax()
    drawdown   = (equity_curve - peak) / peak

    tier_values = np.zeros(len(drawdown), dtype=int)

    tier_values[drawdown <= TIER_THRESHOLDS[RiskTier.CAUTION]]   = int(RiskTier.CAUTION)
    tier_values[drawdown <= TIER_THRESHOLDS[RiskTier.DEFENSIVE]]  = int(RiskTier.DEFENSIVE)
    tier_values[drawdown <= TIER_THRESHOLDS[RiskTier.HALT]]       = int(RiskTier.HALT)
    tier_values[drawdown <= TIER_THRESHOLDS[RiskTier.EMERGENCY]]  = int(RiskTier.EMERGENCY)

    daily_loss_flag = (daily_returns < DAILY_LOSS_LIMIT).shift(1).fillna(False)
    tier_values     = np.where(daily_loss_flag, np.maximum(tier_values, int(RiskTier.HALT)), tier_values)

    size_mult = pd.Series(
        [TIER_SIZE_MULT[RiskTier(t)] for t in tier_values],
        index=equity_curve.index,
    )
    tier_series = pd.Series(tier_values, index=equity_curve.index)

    return size_mult, tier_series

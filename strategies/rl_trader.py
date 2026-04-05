"""
Reinforcement Learning Strategy — PPO on Pairs Spread

Formulates pairs trading as a Markov Decision Process (MDP) and trains an
agent using Proximal Policy Optimization (PPO).

Why PPO over DQN:
  - PPO is an on-policy actor-critic method that clips the policy update,
    preventing large destabilizing steps. For trading environments where the
    reward signal is noisy (market randomness), PPO's conservative updates
    produce more stable learned policies.
  - Empirical comparison: PPO achieved Sharpe 1.75 vs DQN's 1.12 on trading
    tasks (Pichka, 2023). The difference is most pronounced in volatile regimes.
  - DQN requires a discrete action space; PPO supports both discrete and
    continuous actions, making future extensions (fractional sizing) natural.

MDP formulation:
  State  S_t: [z_score_t, spread_vol_t, position_t, vix_t, z_velocity_t,
               days_in_trade_t, rolling_pnl_t]
  Action A_t: {0 = flat, 1 = long spread, 2 = short spread}
  Reward R_t: daily net P&L after transaction costs, normalized by portfolio equity
  Episode:    one rolling window of trading days (252 days), reset each episode

The RL agent does not replace the cointegration filter — it operates on top of
a confirmed cointegrated pair. Its job is to decide WHEN within a cointegrated
regime to enter and exit, using richer contextual information than a static
z-score threshold allows.

Dependencies: gymnasium, stable-baselines3
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import config


class PairsTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for pairs trading on a single confirmed pair.

    State vector (7 features, all normalized to approximately [-1, 1] range):
      0: z_score          — current spread z-score (clipped to [-4, 4])
      1: spread_vol       — 20-day realized volatility of spread returns
      2: position         — current position {-1=short, 0=flat, 1=long}
      3: vix_normalized   — VIX / 30 (rough normalization)
      4: z_velocity       — 1-day change in z-score
      5: days_in_trade    — days current position has been held / 60
      6: rolling_pnl      — 5-day rolling P&L normalized by initial equity

    Actions:
      0 = go flat (close any open position)
      1 = long spread (long y, short x)
      2 = short spread (short y, long x)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        zscore:      pd.Series,
        spread:      pd.Series,
        vix:         pd.Series,
        window:      int   = 252,
        cost_bps:    float = 14.0,  # round-trip in basis points
        initial_equity: float = 100_000,
    ):
        super().__init__()

        self.zscore   = zscore.values.astype(np.float32)
        self.spread   = spread.values.astype(np.float32)
        self.vix      = vix.values.astype(np.float32)
        self.window   = window
        self.cost_bps = cost_bps / 10_000
        self.initial_equity = initial_equity

        # 7 continuous state features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # 3 discrete actions: flat, long, short
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.current_step  = 0
        self.position      = 0       # {-1, 0, 1}
        self.equity        = self.initial_equity
        self.days_in_trade = 0
        self.pnl_history   = [0.0] * 5
        self.entry_z       = None

    def _get_observation(self, t: int) -> np.ndarray:
        z     = self.zscore[t]
        z_vel = self.zscore[t] - self.zscore[t - 1] if t > 0 else 0.0

        # compute rolling spread vol over 20 days
        start = max(0, t - 20)
        spread_slice = self.spread[start: t + 1]
        if len(spread_slice) > 1:
            spread_ret = np.diff(spread_slice) / (spread_slice[:-1] + 1e-8)
            spread_vol = spread_ret.std()
        else:
            spread_vol = 0.0

        vix_norm    = self.vix[t] / 30.0
        days_norm   = self.days_in_trade / 60.0
        rolling_pnl = sum(self.pnl_history) / self.initial_equity

        obs = np.array([
            np.clip(z, -4, 4),
            spread_vol,
            float(self.position),
            vix_norm,
            np.clip(z_vel, -2, 2),
            days_norm,
            rolling_pnl,
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, action: int, t: int) -> float:
        """
        Reward = daily net P&L on the spread position, normalized by equity.

        Spread return for a long-spread position:
          r_spread = (spread[t] - spread[t-1]) / spread[t-1]

        Transaction cost is applied when the position changes.
        """
        if t == 0:
            return 0.0

        # position before applying this action
        prev_position = self.position

        # spread return
        prev_spread   = self.spread[t - 1]
        curr_spread   = self.spread[t]
        if abs(prev_spread) < 1e-6:
            spread_ret = 0.0
        else:
            spread_ret = (curr_spread - prev_spread) / abs(prev_spread)

        # convert action to target position
        target = {0: 0, 1: 1, 2: -1}[action]

        # P&L from prior position
        gross_pnl = prev_position * spread_ret * self.equity

        # transaction cost on position change
        cost = 0.0
        if target != prev_position:
            cost = self.cost_bps * self.equity

        net_pnl = gross_pnl - cost
        reward  = net_pnl / self.initial_equity  # normalize

        # update position tracking
        if target != prev_position:
            self.days_in_trade = 0
        elif target != 0:
            self.days_in_trade += 1

        # update equity and P&L history
        self.equity += net_pnl
        self.pnl_history.pop(0)
        self.pnl_history.append(net_pnl)
        self.position = target

        return float(reward)

    def step(self, action: int):
        t = self.window_start + self.current_step

        reward = self._compute_reward(action, t)
        self.current_step += 1

        done      = self.current_step >= self.window
        truncated = False

        if done:
            obs = self._get_observation(t)
        else:
            obs = self._get_observation(t + 1)

        info = {
            "equity":   self.equity,
            "position": self.position,
        }

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # randomly sample a starting point in the time series
        max_start = max(0, len(self.zscore) - self.window - 1)
        self.window_start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        self._reset_state()

        obs = self._get_observation(self.window_start)
        return obs, {}


def build_and_train_agent(
    zscore:   pd.Series,
    spread:   pd.Series,
    vix:      pd.Series,
    total_timesteps: int = 100_000,
    model_path:      str = "models/ppo_pairs_agent",
) -> PPO:
    """
    Creates the PPO agent, validates the environment, and trains it.

    Architecture: MlpPolicy (multi-layer perceptron) with 2 hidden layers
    of 64 neurons each. For 7 features and 3 actions, this is sufficient.
    Larger networks tend to overfit on noisy financial data.

    Key PPO hyperparameters:
      learning_rate: 3e-4 — standard Adam default, stable for financial tasks
      n_steps:       2048 — steps per rollout before gradient update
      clip_range:    0.2  — maximum policy change per step (the "proximal" part)
      ent_coef:      0.01 — entropy bonus encourages exploration (prevents
                            agent from locking into suboptimal policy early)
    """
    env = PairsTradingEnv(zscore=zscore, spread=spread, vix=vix)

    # validate environment conforms to gymnasium API
    check_env(env, warn=True)

    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = 3e-4,
        n_steps        = 2048,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        policy_kwargs  = dict(net_arch=[64, 64]),
        verbose        = 1,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)

    print(f"  PPO agent trained and saved to {model_path}")
    return model


def evaluate_agent(
    model:    PPO,
    zscore:   pd.Series,
    spread:   pd.Series,
    vix:      pd.Series,
    n_episodes: int = 10,
) -> dict:
    """
    Runs the trained agent on the full dataset (one contiguous episode)
    and computes performance metrics.
    """
    env = PairsTradingEnv(
        zscore=zscore, spread=spread, vix=vix, window=len(zscore) - 1
    )
    env.window_start = 0

    obs, _ = env.reset()
    equity_curve = [env.initial_equity]
    positions    = []
    done         = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        equity_curve.append(info["equity"])
        positions.append(info["position"])

    equity  = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    mdd    = ((np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)).max()
    total_return = (equity[-1] - equity[0]) / equity[0]

    result = {
        "total_return": round(total_return, 4),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(mdd, 4),
        "final_equity": round(equity[-1], 2),
    }

    print(f"  RL agent evaluation:")
    print(f"    total return:  {total_return:.2%}")
    print(f"    sharpe:        {sharpe:.3f}")
    print(f"    max drawdown:  {mdd:.2%}")

    return result

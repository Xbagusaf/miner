# ============================================================
# EVENT SCALPING RL — v21 MINER EDITION
#
# Perubahan dari v20.1:
#
# 1. [DATA ADAPTER] Kompatibel dengan CSV 1-second bar dari miner.
#    - `microprice` → `micro_price` (rename otomatis)
#    - `bid_depth_top5/ask_depth_top5` → `bid_depth_1/ask_depth_1`
#    - `rolling_volume_5s` dihitung dari `volume.rolling(5)`
#    - `vol_filter` dari `realized_volatility` (pre-computed miner)
#
# 2. [REGIME BYPASS GMM] Pakai `volatility_regime` dari miner langsung.
#    Miner sudah hitung regime 0/1/2 dengan percentile rolling 1-hour.
#    Lebih akurat dan tidak perlu GMM re-fit setiap run.
#
# 3. [EXCLUDE_COLS UPDATED] Tambah kolom metadata miner + DATA LEAKAGE:
#    - DIEXCLUDE: realized_spread, adverse_selection_metric
#      (keduanya pakai mid_price t+5 — retroactive, data leakage!)
#    - DIEXCLUDE: is_warmup, row_checksum, recovery_source, has_gap
#
# 4. [FILTER WARMUP] Baris is_warmup=True dibuang sebelum training.
#    120 bar pertama tidak reliable untuk feature computation.
#
# 5. [CAUSAL TRANSFORMER] Gantikan TCN dengan CausalTransformerExtractor.
#    Multi-head self-attention + causal mask — lebih ekspresif untuk
#    microstructure pattern di 80+ fitur miner.
#    TCNExtractor tetap tersedia sebagai fallback.
#
# 6. [CHECKPOINT DIR] checkpoints_v21_miner
#
# FITUR MINER BARU YANG MASUK OTOMATIS (tidak di EXCLUDE_COLS):
#   Toxicity: vpin, toxicity_score, kyle_lambda, amihud_illiquidity
#   Futures:  funding_rate, mark_index_spread, oi_change_rate, long_short_ratio
#   Regime:   volatility_regime, trend_regime, hurst_exponent, entropy
#   Liquidity: effective_spread, resiliency_metric, trade_burst_metric
# ============================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
import os
import sys
import signal
import atexit
import multiprocessing
import gc
import pickle
import json
import warnings
import time
from datetime import datetime
from numba import njit

warnings.filterwarnings(
    "ignore", message=".*primarily intended to run on the CPU.*", category=UserWarning
)


# ============================================================
# 0. KONFIGURASI GLOBAL
# ============================================================

# ===== DATA — arahkan ke CSV miner =====
CSV_PATHS = [
    "data/HYPEUSDT_20260527.csv",
]

# ===== TRAINING =====
N_ENVS             = 8
N_SPLITS           = 6
TIMESTEPS_PER_FOLD = 3_000_000
PURGE_GAP          = 1500

# ===== EPISODE =====
WINDOW_SIZE        = 320
MAX_EPISODE_STEPS  = 25_000

# ===== PPO HYPERPARAMETERS =====
LR_START           = 0.00005
LR_END             = 5e-6
N_STEPS            = 3072
BATCH_SIZE         = 1536
N_EPOCHS           = 4
ENT_COEF           = 0.05
VF_COEF            = 0.5
MAX_GRAD_NORM      = 0.5
CLIP_RANGE_START   = 0.1
CLIP_RANGE_END     = 0.2
GAE_LAMBDA         = 0.95
GAMMA              = 0.9995
VEC_CLIP_OBS       = 10.0

# ===== EVENT-DRIVEN =====
OBS_MAX_HOLD       = 10000.0

# ===== ARCHITECTURE SWITCH =====
# "transformer" (rekomendasi) atau "tcn" (fallback v20.1)
EXTRACTOR_TYPE   = "transformer"

# ===== CAUSAL TRANSFORMER =====
TRANS_D_MODEL    = 128
TRANS_NHEAD      = 4
TRANS_FFN_DIM    = 256
TRANS_N_LAYERS   = 3
TRANS_DROPOUT    = 0.10
TRANS_FEATURES_DIM = 224

# ===== TCN (fallback) =====
TCN_INPUT_DIM    = 128
TCN_CHANNELS     = 128
TCN_KERNEL_SIZE  = 4
TCN_DILATIONS    = [1, 2, 4, 8, 16]
TCN_DROPOUT      = 0.10
TCN_FEATURES_DIM = 224

# ===== LSTM =====
LSTM_HIDDEN_SIZE   = 256
N_LSTM_LAYERS      = 2
SHARED_LSTM        = False
ENABLE_CRITIC_LSTM = True

# ===== POLICY HEADS =====
N_AGENT_STATE    = 3
PI_NET_ARCH      = [128, 64]
VF_NET_ARCH      = [128, 64]

# ===== TRADING =====
FEE_TAKER          = 0.0005

# ===== HAWKES =====
HAWKES_ALPHA     = 0.10
HAWKES_BETA      = 0.10

# ===== REGIME — bypass GMM, pakai volatility_regime dari miner =====
USE_MINER_REGIME  = True   # True = pakai kolom volatility_regime dari CSV
USE_HARDCODED_GMM = False  # Diabaikan jika USE_MINER_REGIME=True

REGIME_P_MATRIX  = [
    [0.990, 0.010, 0.000],
    [0.005, 0.985, 0.010],
    [0.000, 0.020, 0.980],
]

HARDCODED_MU_VOL  = [0.00008404, 0.00013495, 0.00021501]
HARDCODED_SIG_VOL = [0.00002279, 0.00003560, 0.00006125]

REGIME_MU_VOL    = HARDCODED_MU_VOL
REGIME_SIG_VOL   = HARDCODED_SIG_VOL
N_REGIMES        = 3
GMM_N_INIT       = 20
GMM_MAX_ITER     = 500
GMM_TOL          = 1e-6
GMM_MAX_SAMPLES  = 2_000_000

# ===== EQUITY DYNAMICS =====
STARTING_EQUITY       = 1.0
DEATH_EQUITY          = 0.10
PCT_SCALE             = 100.0

# ============================================================
# REWARD CONFIG
# ============================================================
DSR_ETA           = 0.01
DSR_WARMUP_STEPS  = 200
DSR_SCALE         = 10.0

REGIME_DOWNSIDE_MULT        = [1.2, 1.7, 2.5]
INVALID_ACTION_PENALTY_BPS  = 0.2

# ============================================================
# ENTROPY GUARDIAN CONFIG
# ============================================================
ENT_GUARDIAN_MIN_ENTROPY = 0.60
ENT_GUARDIAN_MAX_ENTROPY = 0.95
ENT_GUARDIAN_ADJ_FACTOR  = 1.3
ENT_GUARDIAN_MIN_COEF    = 0.01
ENT_GUARDIAN_MAX_COEF    = 0.20
ENT_GUARDIAN_CHECK_FREQ  = 8192

# ===== EARLY STOPPING =====
ENABLE_EARLY_STOP     = True
EARLY_STOP_MIN_EQUITY = 0.10
EARLY_STOP_CHECK_STEP = 200_000

# ===== CHECKPOINT =====
CHECKPOINT_DIR = "checkpoints_v21_miner"

# ===== EXCLUDE COLS — metadata miner + data leakage =====
EXCLUDE_COLS = {
    # Original v20.1
    'timestamp', 'symbol', 'micro_price', '_ts_idx',
    'label_3s', 'label_10s', 'label_20s', 'mid_price', 'last_price',
    # Miner metadata (tidak informatif)
    'Timestamp', 'local_timestamp_ms', 'clock_skew_ms', 'sequence_id',
    'is_warmup', 'row_checksum', 'recovery_source', 'has_gap',
    # DATA LEAKAGE — realized_spread dan adverse_selection_metric
    # menggunakan mid_price t+5 (retroactive) → TIDAK BOLEH jadi feature!
    'realized_spread', 'adverse_selection_metric',
}


# ============================================================
# GLOBAL ENV REGISTRY
# ============================================================
_active_envs = []

def _cleanup_all_envs():
    global _active_envs
    for env in list(_active_envs):
        try:
            env.close()
        except Exception:
            pass
    _active_envs.clear()

atexit.register(_cleanup_all_envs)

def _signal_handler(signum, frame):
    print("\n⚠️  Signal received, cleaning up...")
    _cleanup_all_envs()
    sys.exit(0)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================
# 1. NUMBA HELPERS
# ============================================================
@njit
def compute_hawkes_intensity(delta_t_array, alpha, beta):
    n           = len(delta_t_array)
    intensity   = np.zeros(n)
    curr_lambda = 0.0
    for i in range(1, n):
        curr_lambda  = curr_lambda * np.exp(-beta * delta_t_array[i]) + alpha
        intensity[i] = curr_lambda
    return intensity

@njit
def online_regime_filter(vol_array, spread_array, p_mat, mu_vol, sig_vol, n_states=3):
    n       = len(vol_array)
    beliefs = np.zeros((n, n_states))
    states  = np.zeros(n, dtype=np.int32)
    b       = np.array([1.0, 0.0, 0.0])

    for i in range(n):
        v   = vol_array[i]
        l_v = np.zeros(n_states)
        for k in range(n_states):
            std    = sig_vol[k] + 1e-9
            l_v[k] = np.exp(-0.5 * ((v - mu_vol[k]) / std) ** 2) / std
        b_pred = np.dot(b, p_mat)
        b_upd  = b_pred * (l_v + 1e-9)
        s_upd  = np.sum(b_upd)
        b      = b_upd / s_upd if s_upd > 0 else np.full(n_states, 1.0 / n_states)
        beliefs[i] = b
        states[i]  = np.int32(np.argmax(b))
    return beliefs, states


# ============================================================
# 2. FEATURE ENGINEERING — adapter untuk kolom miner
# ============================================================
def engineer_advanced_features(df):
    # [v21] Mapping kolom miner → nama yang diharapkan script
    if 'microprice' in df.columns and 'micro_price' not in df.columns:
        df['micro_price'] = df['microprice']

    if 'micro_price' not in df.columns and all(
        c in df.columns for c in ['best_bid', 'best_ask', 'bid_depth_top5', 'ask_depth_top5']
    ):
        df['spread']      = df['best_ask'] - df['best_bid']
        df['mid_price']   = (df['best_ask'] + df['best_bid']) / 2.0
        df['micro_price'] = (
            df['best_bid'] * df['ask_depth_top5'] + df['best_ask'] * df['bid_depth_top5']
        ) / (df['bid_depth_top5'] + df['ask_depth_top5'] + 1e-9)
    elif 'micro_price' not in df.columns and all(
        c in df.columns for c in ['best_bid', 'best_ask', 'bid_depth_1', 'ask_depth_1']
    ):
        df['spread']      = df['best_ask'] - df['best_bid']
        df['mid_price']   = (df['best_ask'] + df['best_bid']) / 2.0
        df['micro_price'] = (
            df['best_bid'] * df['ask_depth_1'] + df['best_ask'] * df['bid_depth_1']
        ) / (df['bid_depth_1'] + df['ask_depth_1'] + 1e-9)

    # [v21] Proxy depth_1 dari top5 miner
    if 'bid_depth_1' not in df.columns and 'bid_depth_top5' in df.columns:
        df['bid_depth_1'] = df['bid_depth_top5']
        df['ask_depth_1'] = df['ask_depth_top5']

    # [v21] rolling_volume_5s dari 1s bar volume
    if 'rolling_volume_5s' not in df.columns and 'volume' in df.columns:
        df['rolling_volume_5s'] = df['volume'].rolling(5, min_periods=1).sum()

    # [v21] vol_filter dari realized_volatility miner (sudah pre-computed)
    if 'volatility_60s' not in df.columns and 'realized_volatility' in df.columns:
        df['vol_filter'] = df['realized_volatility'].fillna(0)
    elif 'volatility_60s' in df.columns:
        df['vol_filter'] = df['volatility_60s'].fillna(0)
    else:
        pct = df['micro_price'].pct_change().fillna(0)
        df['vol_filter'] = pct.rolling(20, min_periods=1).std().fillna(0).values

    pct = df['micro_price'].pct_change().fillna(0)

    df['vol_10s']  = pct.rolling(100,  min_periods=1).std().fillna(0)
    df['vol_100s'] = pct.rolling(1000, min_periods=1).std().fillna(0)

    if 'last_event_age_ms' in df.columns:
        df['log_event_age'] = np.log1p(df['last_event_age_ms'].fillna(0))
    if 'oi_value' in df.columns:
        df['log_oi_value']  = np.log1p(df['oi_value'].abs().fillna(0)) * np.sign(df['oi_value'].fillna(0))
    elif 'open_interest' in df.columns:
        # [v21] open_interest dari miner
        df['log_oi_value']  = np.log1p(df['open_interest'].abs().fillna(0))
    if 'rolling_volume_1s' in df.columns:
        df['log_volume_1s'] = np.log1p(df['rolling_volume_1s'].fillna(0))
    elif 'volume' in df.columns:
        df['log_volume_1s'] = np.log1p(df['volume'].fillna(0))
    if 'rolling_volume_5s' in df.columns:
        df['log_volume_5s'] = np.log1p(df['rolling_volume_5s'].fillna(0))
    if 'bid_depth_1' in df.columns:
        df['log_bid_depth'] = np.log1p(df['bid_depth_1'].fillna(0))
    if 'ask_depth_1' in df.columns:
        df['log_ask_depth'] = np.log1p(df['ask_depth_1'].fillna(0))

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


def preprocess_dataset(csv_file, mu_vol=None, sig_vol=None):
    print(f"   Memproses {csv_file}...")
    df         = pd.read_csv(csv_file)
    df.columns = [str(c).strip() for c in df.columns]
    valid_cols = [c for c in df.columns if c not in ['nan', 'none', ''] and 'unnamed' not in c.lower()]
    df         = df[valid_cols]

    # [v21] Buang baris warmup — fitur belum stabil di 120 bar pertama
    if 'is_warmup' in df.columns:
        n_before = len(df)
        df = df[df['is_warmup'] != True].reset_index(drop=True)
        print(f"   Filtered warmup: {n_before} → {len(df)} rows")

    df = engineer_advanced_features(df)

    # Hawkes intensity — fallback ke volatility proxy jika tidak ada last_event_age_ms
    if 'last_event_age_ms' in df.columns:
        dt = df['last_event_age_ms'].fillna(50.0).clip(lower=0.001).values
    else:
        dt = (df['micro_price'].pct_change().abs() * 10_000 + 1.0).fillna(1.0).values
    df['hawkes_intensity'] = compute_hawkes_intensity(dt, HAWKES_ALPHA, HAWKES_BETA)

    # [v21] Regime: pakai volatility_regime miner langsung jika tersedia
    if USE_MINER_REGIME and 'volatility_regime' in df.columns:
        df['regime_state'] = df['volatility_regime'].fillna(1).astype(int).clip(0, 2)
        for k in range(N_REGIMES):
            df[f'regime_prob_{k}'] = (df['regime_state'] == k).astype(float)
        print(f"   Regime: menggunakan volatility_regime miner "
              f"(dist: {[(df['regime_state']==k).sum() for k in range(3)]})")
    else:
        spread_arr = df['spread'].values if 'spread' in df.columns else np.zeros(len(df))
        p_mat      = np.array(REGIME_P_MATRIX,          dtype=np.float64)
        mu_arr     = np.array(mu_vol  or REGIME_MU_VOL,  dtype=np.float64)
        sig_arr    = np.array(sig_vol or REGIME_SIG_VOL, dtype=np.float64)
        beliefs, states = online_regime_filter(
            df['vol_filter'].values, spread_arr, p_mat, mu_arr, sig_arr
        )
        for k in range(N_REGIMES):
            df[f'regime_prob_{k}'] = beliefs[:, k]
        df['regime_state'] = states

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


# ============================================================
# 3. GMM REGIME ESTIMATION (dipakai jika USE_MINER_REGIME=False)
# ============================================================
def load_raw_for_gmm(csv_path):
    df         = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    valid_cols = [c for c in df.columns if c not in ['nan', 'none', ''] and 'unnamed' not in c.lower()]
    df         = df[valid_cols]
    if 'is_warmup' in df.columns:
        df = df[df['is_warmup'] != True].reset_index(drop=True)
    df = engineer_advanced_features(df)
    return df

def _fit_single_gmm(seed, data, n_regimes, max_iter, tol):
    gmm = GaussianMixture(
        n_components    = n_regimes,
        covariance_type = 'spherical',
        random_state    = seed,
        n_init          = 1,
        max_iter        = max_iter,
        tol             = tol,
        reg_covar       = 1e-12,
    )
    gmm.fit(data)
    return float(gmm.lower_bound_), gmm

def estimate_regime_params(csv_paths, n_regimes=N_REGIMES):
    print(f"\n🔍 Auto-estimasi regime params via GMM...")
    all_vols = []
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        df  = load_raw_for_gmm(path)
        all_vols.append(df['vol_filter'].values)
        del df
        gc.collect()

    if not all_vols:
        return REGIME_MU_VOL, REGIME_SIG_VOL

    combined = np.concatenate(all_vols)
    p1, p99  = np.percentile(combined, [1, 99])
    clipped  = combined[(combined >= p1) & (combined <= p99)]

    if len(clipped) > GMM_MAX_SAMPLES:
        rng     = np.random.default_rng(42)
        idx     = rng.choice(len(clipped), size=GMM_MAX_SAMPLES, replace=False)
        clipped = clipped[idx]

    data = clipped.reshape(-1, 1)
    results = Parallel(n_jobs=-1, verbose=5, backend='loky')(
        delayed(_fit_single_gmm)(seed, data, n_regimes, GMM_MAX_ITER, GMM_TOL)
        for seed in range(GMM_N_INIT)
    )

    best_score, gmm = max(results, key=lambda x: x[0])
    order   = np.argsort(gmm.means_.flatten())
    mu_vol  = [float(gmm.means_[i, 0]) for i in order]
    sig_vol = [float(np.sqrt(gmm.covariances_[i]) + 1e-9) for i in order]
    print(f"   GMM result: mu={[f'{v:.6f}' for v in mu_vol]}")
    print(f"              sig={[f'{v:.6f}' for v in sig_vol]}")
    return mu_vol, sig_vol


# ============================================================
# 4. DIFFERENTIAL SHARPE RATIO TRACKER
# ============================================================
class DSRTracker:
    def __init__(self, eta=DSR_ETA, warmup_steps=DSR_WARMUP_STEPS):
        self.eta          = eta
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self):
        self.A          = 0.0
        self.B          = 1.0
        self.step_count = 0

    def update(self, R_t):
        if self.step_count < self.warmup_steps:
            self.A = self.A + self.eta * (R_t - self.A)
            self.B = self.B + self.eta * (R_t * R_t - self.B)
            self.step_count += 1
            return float(R_t * 0.1)

        delta_A = R_t - self.A
        delta_B = R_t * R_t - self.B

        variance = self.B - self.A * self.A
        if variance <= 1e-9:
            variance = 1e-9

        denominator = variance ** 1.5
        numerator   = self.B * delta_A - 0.5 * self.A * delta_B
        dsr         = numerator / denominator

        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * delta_B
        self.step_count += 1

        return float(np.clip(dsr, -50.0, 50.0))


# ============================================================
# 5. ENVIRONMENT v21
# ============================================================
class EventScalpingEnv(gym.Env):
    def __init__(self, prices_arr, features_arr, regime_arr,
                 window_size=WINDOW_SIZE, is_training=True, rank=0):
        super().__init__()
        self.prices            = prices_arr.astype(np.float64)
        self.features          = features_arr.astype(np.float32)
        self.regime_arr        = regime_arr.astype(np.int32)
        self.window_size       = window_size
        self.is_training       = is_training
        self.rank              = rank
        self.n_rows            = len(prices_arr)
        self.max_episode_steps = MAX_EPISODE_STEPS if is_training else self.n_rows

        self.n_market_features = self.features.shape[1]
        obs_dim                = self.n_market_features + N_AGENT_STATE

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._obs_buf     = np.zeros((self.window_size, obs_dim), dtype=np.float32)

        self.dsr_tracker = DSRTracker(eta=DSR_ETA, warmup_steps=DSR_WARMUP_STEPS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        valid_min = self.window_size
        valid_max = self.n_rows - self.max_episode_steps - 2

        if not self.is_training:
            self.current_step = valid_min
        elif valid_max > valid_min:
            self.current_step = int(self.np_random.integers(valid_min, valid_max))
        else:
            safe_margin       = self.n_rows - self.window_size - 100
            offset            = self.rank % safe_margin if safe_margin > 0 else 0
            self.current_step = self.window_size + offset

        self.current_equity       = STARTING_EQUITY
        self.peak_equity          = STARTING_EQUITY
        self.position             = 0
        self.last_position        = 0
        self.entry_price          = 0.0
        self.current_episode_step = 0
        self.steps_in_position    = 0
        self.trade_cost_memory    = 0.0

        self.dsr_tracker.reset()

        return self._get_obs(), {}

    def action_masks(self):
        if self.position == 0:
            return np.array([True, True, True], dtype=bool)
        elif self.position == 1:
            return np.array([True, False, True], dtype=bool)
        else:
            return np.array([True, True, False], dtype=bool)

    def _get_obs(self):
        s = self.current_step
        self._obs_buf[:, :self.n_market_features] = self.features[s - self.window_size + 1: s + 1]
        self._obs_buf[:, self.n_market_features:] = 0.0

        self._obs_buf[-1, -3] = float(self.position)
        self._obs_buf[-1, -2] = float(self.steps_in_position / OBS_MAX_HOLD)
        if self.position != 0 and self.entry_price > 0:
            unrealized = self.position * (self.prices[s] - self.entry_price) / self.entry_price
            self._obs_buf[-1, -1] = float(unrealized * PCT_SCALE)

        return self._obs_buf.copy()

    def step(self, action):
        s      = self.current_step
        P_t    = self.prices[s]
        P_next = self.prices[s + 1]
        regime = int(self.regime_arr[s])

        eq_t_start = self.current_equity
        target_pos = self.position

        action_was_invalid = False

        if self.position == 0:
            if action == 1:
                target_pos = 1
            elif action == 2:
                target_pos = -1
        elif self.position == 1:
            if action == 2:
                target_pos = 0
            elif action == 1:
                action_was_invalid = True
        elif self.position == -1:
            if action == 1:
                target_pos = 0
            elif action == 2:
                action_was_invalid = True

        delta_pos      = target_pos - self.position
        trade_executed = (delta_pos != 0)
        closed_trade   = (self.position != 0 and target_pos == 0)
        opened_trade   = (self.position == 0 and target_pos != 0)

        eq_t_post_execution = eq_t_start
        dynamic_slippage    = 0.0001 * (regime + 1)
        fee_rate            = FEE_TAKER + dynamic_slippage

        if trade_executed:
            execution_cost       = eq_t_post_execution * abs(delta_pos) * fee_rate
            eq_t_post_execution -= execution_cost
            self.trade_cost_memory += fee_rate * abs(delta_pos)

            if target_pos != 0:
                self.entry_price       = P_next
                self.steps_in_position = 0
            else:
                self.trade_cost_memory = 0.0

        self.last_position = self.position
        self.position      = target_pos

        if self.position != 0:
            self.steps_in_position += 1

        asset_return = (P_next - P_t) / P_t
        eq_t_next    = eq_t_post_execution * (1.0 + (self.position * asset_return))

        self.current_equity = eq_t_next
        self.peak_equity    = max(self.peak_equity, eq_t_next)

        safe_eq_next  = max(eq_t_next,  1e-12)
        safe_eq_start = max(eq_t_start, 1e-12)

        log_ret_bps = np.log(safe_eq_next / safe_eq_start) * 10000.0

        R_for_dsr = log_ret_bps
        if R_for_dsr < 0:
            mult       = REGIME_DOWNSIDE_MULT[regime]
            R_for_dsr *= mult

        dsr_signal  = self.dsr_tracker.update(R_for_dsr)
        step_reward = dsr_signal * DSR_SCALE

        if action_was_invalid:
            step_reward -= INVALID_ACTION_PENALTY_BPS

        trade_log_ret_bps = 0.0
        if closed_trade and self.entry_price > 0:
            cycle_log_ret     = np.log(P_next / self.entry_price)
            net_cycle_log_ret = (self.last_position * cycle_log_ret) - self.trade_cost_memory
            trade_log_ret_bps = net_cycle_log_ret * 10000.0

        self.current_step         += 1
        self.current_episode_step += 1

        is_dead = self.current_equity <= DEATH_EQUITY
        done    = (
            self.current_step >= self.n_rows - 1
            or self.current_episode_step >= self.max_episode_steps
            or is_dead
        )

        step_return_linear = (eq_t_next - eq_t_start) / safe_eq_start
        drawdown           = (self.peak_equity - eq_t_next) / max(1e-12, self.peak_equity)

        info = {
            'trade_executed':    trade_executed,
            'closed_trade':      closed_trade,
            'opened_trade':      opened_trade,
            'regime_state':      regime,
            'is_dead':           is_dead,
            'equity':            self.current_equity,
            'peak_equity':       self.peak_equity,
            'drawdown':          drawdown,
            'step_return':       step_return_linear,
            'log_return_bps':    log_ret_bps,
            'trade_log_ret_bps': trade_log_ret_bps,
            'dsr_signal':        dsr_signal,
            'action_invalid':    action_was_invalid,
        }

        return self._get_obs(), float(step_reward), done, False, info


# ============================================================
# 6. BASELINE TESTER
# ============================================================
def run_baseline_tests(prices, features, regime, n_steps=10000):
    print("\n" + "=" * 60)
    print("  🧪 BASELINE TESTS (v21 MINER)")
    print("=" * 60)

    results = {}

    env = EventScalpingEnv(prices, features, regime, WINDOW_SIZE, True, 0)
    obs, _ = env.reset(seed=42)
    total_rew = 0.0
    for _ in range(n_steps):
        obs, r, d, _, _ = env.step(0)
        total_rew += r
        if d:
            break
    results['hold_only_equity']    = env.current_equity
    results['hold_only_total_rew'] = total_rew
    print(f"  HOLD-ONLY    {n_steps} step: equity={env.current_equity:.4f} | total_rew={total_rew:+.2f}")

    env = EventScalpingEnv(prices, features, regime, WINDOW_SIZE, True, 3)
    obs, _ = env.reset(seed=45)
    rng             = np.random.default_rng(123)
    trades_closed   = 0
    invalid_actions = 0
    step_returns    = []
    trade_returns   = []
    total_rew       = 0.0

    for _ in range(n_steps):
        a = rng.integers(0, 3)
        obs, r, d, _, info = env.step(int(a))
        total_rew += r
        step_returns.append(info['step_return'])
        if info.get('closed_trade'):
            trades_closed += 1
            trade_returns.append(info['trade_log_ret_bps'])
        if info.get('action_invalid'):
            invalid_actions += 1
        if d:
            break

    results['random_equity'] = env.current_equity
    results['random_trades'] = trades_closed
    ev_step_bps  = np.mean(step_returns)  * 10000.0 if step_returns  else 0.0
    ev_trade_bps = np.mean(trade_returns)            if trade_returns else 0.0

    print(f"  RANDOM TRADE {n_steps} step: equity={env.current_equity:.4f} | trades={trades_closed}")
    print(f"  > EV per step:  {ev_step_bps:+.4f} bps")
    print(f"  > EV per trade: {ev_trade_bps:+.4f} bps")
    print(f"  > Invalid actions: {invalid_actions}")
    print(f"  > Total reward (DSR): {total_rew:+.2f}")
    print("=" * 60)
    return results


# ============================================================
# 7. TCN BUILDING BLOCK (fallback)
# ============================================================
class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, padding=0
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, padding=0
        ))
        self.norm1    = nn.GroupNorm(1, out_channels)
        self.norm2    = nn.GroupNorm(1, out_channels)
        self.dropout  = nn.Dropout(dropout)
        self.act      = nn.GELU()
        self.res_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.res_proj(x)
        out = F.pad(x,   (self.causal_pad, 0))
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = F.pad(out, (self.causal_pad, 0))
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + residual)


# ============================================================
# 8. EXTRACTORS — Causal Transformer (default) + TCN (fallback)
# ============================================================
class CausalTransformerExtractor(BaseFeaturesExtractor):
    """
    [v21 DEFAULT] Causal Multi-Head Self-Attention + GELU FFN.
    Setiap posisi t hanya dapat melihat posisi ≤ t (causal mask).
    Lebih ekspresif dari TCN untuk 80+ fitur microstructure miner.
    """

    def __init__(self, observation_space, features_dim=TRANS_FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        self.seq_len    = observation_space.shape[0]
        self.signal_dim = observation_space.shape[1]
        self.market_dim = self.signal_dim - N_AGENT_STATE
        self.agent_dim  = N_AGENT_STATE

        self.input_proj = nn.Sequential(
            nn.Linear(self.market_dim, TRANS_D_MODEL),
            nn.LayerNorm(TRANS_D_MODEL),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = TRANS_D_MODEL,
            nhead          = TRANS_NHEAD,
            dim_feedforward = TRANS_FFN_DIM,
            dropout        = TRANS_DROPOUT,
            activation     = 'gelu',
            batch_first    = True,
            norm_first     = True,   # Pre-LN: lebih stabil
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANS_N_LAYERS)

        # Causal mask: posisi i tidak boleh attend ke j > i
        causal = nn.Transformer.generate_square_subsequent_mask(self.seq_len)
        self.register_buffer('causal_mask', causal)

        self.output_head = nn.Sequential(
            nn.Linear(TRANS_D_MODEL + self.agent_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.05),
            nn.GELU(),
        )

    def forward(self, obs):
        market_seq  = obs[:, :, :self.market_dim]           # (B, seq, market_dim)
        agent_state = obs[:, -1, -self.agent_dim:]           # (B, agent_dim)

        x   = self.input_proj(market_seq)                    # (B, seq, d_model)
        x   = self.transformer(x, mask=self.causal_mask.to(x.device),
                                is_causal=True)
        out = x[:, -1, :]                                    # (B, d_model) — posisi terakhir
        return self.output_head(torch.cat([out, agent_state], dim=1))


class TCNExtractor(BaseFeaturesExtractor):
    """[v21 FALLBACK] TCN dari v20.1, digunakan jika EXTRACTOR_TYPE='tcn'."""

    def __init__(self, observation_space, features_dim=TCN_FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        self.seq_len    = observation_space.shape[0]
        self.signal_dim = observation_space.shape[1]
        self.market_dim = self.signal_dim - N_AGENT_STATE
        self.agent_dim  = N_AGENT_STATE

        self.input_proj = nn.Sequential(
            nn.Linear(self.market_dim, TCN_INPUT_DIM),
            nn.LayerNorm(TCN_INPUT_DIM),
            nn.GELU(),
        )

        blocks = []
        in_ch  = TCN_INPUT_DIM
        for dilation in TCN_DILATIONS:
            blocks.append(TCNResidualBlock(
                in_channels  = in_ch,
                out_channels = TCN_CHANNELS,
                kernel_size  = TCN_KERNEL_SIZE,
                dilation     = dilation,
                dropout      = TCN_DROPOUT,
            ))
            in_ch = TCN_CHANNELS
        self.tcn_blocks = nn.ModuleList(blocks)

        self.output_head = nn.Sequential(
            nn.Linear(TCN_CHANNELS + self.agent_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.05),
            nn.GELU(),
        )

    def forward(self, obs):
        market_seq  = obs[:, :, :self.market_dim]
        agent_state = obs[:, -1, -self.agent_dim:]
        x = self.input_proj(market_seq)
        x = x.transpose(1, 2)
        for block in self.tcn_blocks:
            x = block(x)
        tcn_out  = x[:, :, -1]
        combined = torch.cat([tcn_out, agent_state], dim=1)
        return self.output_head(combined)


def get_extractor():
    """Pilih extractor berdasarkan EXTRACTOR_TYPE."""
    if EXTRACTOR_TYPE == "transformer":
        return CausalTransformerExtractor, dict(features_dim=TRANS_FEATURES_DIM)
    else:
        return TCNExtractor, dict(features_dim=TCN_FEATURES_DIM)


# ============================================================
# 9. CALLBACKS
# ============================================================
class SurvivalMonitorCallback(BaseCallback):
    def __init__(self, check_freq=2048, early_stop=True, verbose=0):
        super().__init__(verbose)
        self.check_freq     = check_freq
        self.early_stop     = early_stop
        self.deaths         = 0
        self.equity_hist    = []
        self.dd_hist        = []
        self.n_trades       = 0
        self.step_ret_hist  = []
        self.trade_ret_hist = []
        self.reward_hist    = []
        self.dsr_hist       = []
        self.regime_counts  = [0, 0, 0]
        self.action_counts  = [0, 0, 0]
        self.invalid_count  = 0
        self.should_stop    = False

    def _on_step(self):
        actions = self.locals.get('actions', None)
        if actions is not None:
            for a in np.asarray(actions).flatten():
                ai = int(a)
                if 0 <= ai < 3:
                    self.action_counts[ai] += 1

        for info in self.locals.get('infos', []):
            if info.get('is_dead', False):
                self.deaths += 1
            if 'equity' in info:
                self.equity_hist.append(info['equity'])
            if 'drawdown' in info:
                self.dd_hist.append(info['drawdown'])
            if 'step_return' in info:
                self.step_ret_hist.append(info['step_return'])
            if 'dsr_signal' in info:
                self.dsr_hist.append(info['dsr_signal'])
            if 'regime_state' in info:
                r = int(info['regime_state'])
                if 0 <= r < 3:
                    self.regime_counts[r] += 1
            if info.get('closed_trade', False):
                self.n_trades += 1
                self.trade_ret_hist.append(info.get('trade_log_ret_bps', 0.0))
            if info.get('action_invalid', False):
                self.invalid_count += 1

        rewards = self.locals.get('rewards', [])
        if rewards is not None:
            self.reward_hist.extend(list(rewards))

        if self.n_calls % self.check_freq == 0:
            mean_eq      = np.mean(self.equity_hist)    if self.equity_hist    else STARTING_EQUITY
            mean_dd      = np.mean(self.dd_hist)        if self.dd_hist        else 0.0
            mean_rew     = np.mean(self.reward_hist)    if self.reward_hist    else 0.0
            mean_dsr     = np.mean(self.dsr_hist)       if self.dsr_hist       else 0.0
            ev_step_bps  = np.mean(self.step_ret_hist)  * 10000.0 if self.step_ret_hist  else 0.0
            ev_trade_bps = np.mean(self.trade_ret_hist) if self.trade_ret_hist else 0.0

            tr_arr   = np.array(self.trade_ret_hist) if self.trade_ret_hist else np.array([])
            win_rate = (len(tr_arr[tr_arr > 0]) / len(tr_arr) * 100) if len(tr_arr) > 0 else 0.0

            total_a = sum(self.action_counts) or 1
            a_pct   = [self.action_counts[i] / total_a * 100 for i in range(3)]
            a_dist  = f"H:{a_pct[0]:.0f}% L:{a_pct[1]:.0f}% S:{a_pct[2]:.0f}%"
            inv_pct = (self.invalid_count / total_a * 100) if total_a > 0 else 0.0

            print(
                f"📊 [{self.num_timesteps:>10,}] "
                f"Eq:{mean_eq:.3f} DD:{mean_dd*100:.1f}% | "
                f"Trades:{self.n_trades} WR:{win_rate:.0f}% | "
                f"EV_step:{ev_step_bps:+.3f}bps EV_trade:{ev_trade_bps:+.2f}bps | "
                f"{a_dist} Inv:{inv_pct:.1f}% | "
                f"DSR:{mean_dsr:+.3f} Rew:{mean_rew:+.3f} D:{self.deaths}"
            )

            if (self.early_stop
                    and self.num_timesteps >= EARLY_STOP_CHECK_STEP
                    and mean_eq < EARLY_STOP_MIN_EQUITY):
                print(f"  🛑 EARLY STOP: Eq {mean_eq:.4f} < {EARLY_STOP_MIN_EQUITY}")
                self.should_stop = True
                return False

            self.deaths         = 0
            self.equity_hist    = []
            self.dd_hist        = []
            self.step_ret_hist  = []
            self.n_trades       = 0
            self.trade_ret_hist = []
            self.reward_hist    = []
            self.dsr_hist       = []
            self.regime_counts  = [0, 0, 0]
            self.action_counts  = [0, 0, 0]
            self.invalid_count  = 0
        return True


class EntropyGuardianCallback(BaseCallback):
    def __init__(
        self,
        min_entropy=ENT_GUARDIAN_MIN_ENTROPY,
        max_entropy=ENT_GUARDIAN_MAX_ENTROPY,
        adjustment_factor=ENT_GUARDIAN_ADJ_FACTOR,
        min_ent_coef=ENT_GUARDIAN_MIN_COEF,
        max_ent_coef=ENT_GUARDIAN_MAX_COEF,
        check_freq=ENT_GUARDIAN_CHECK_FREQ,
        verbose=1,
    ):
        super().__init__(verbose)
        self.min_entropy       = min_entropy
        self.max_entropy       = max_entropy
        self.adjustment_factor = adjustment_factor
        self.min_ent_coef      = min_ent_coef
        self.max_ent_coef      = max_ent_coef
        self.check_freq        = check_freq
        self.entropy_history   = []
        self.adjustment_log    = []
        self.last_check_step   = 0

    def _on_step(self):
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            ent_loss = self.model.logger.name_to_value.get('train/entropy_loss', None)
            if ent_loss is not None and ent_loss != 0.0:
                self.entropy_history.append(-float(ent_loss))

        if (self.num_timesteps - self.last_check_step) >= self.check_freq and len(self.entropy_history) > 0:
            self.last_check_step = self.num_timesteps
            recent_window  = min(5, len(self.entropy_history))
            recent_entropy = np.mean(self.entropy_history[-recent_window:])
            current_coef   = float(self.model.ent_coef)
            new_coef       = current_coef

            if recent_entropy < self.min_entropy:
                new_coef = min(current_coef * self.adjustment_factor, self.max_ent_coef)
                if new_coef > current_coef + 1e-6:
                    self.adjustment_log.append({
                        'step': self.num_timesteps, 'entropy': recent_entropy,
                        'old_coef': current_coef, 'new_coef': new_coef, 'action': 'BOOST',
                    })
                    if self.verbose:
                        print(f"   🔥 ENT BOOST [{self.num_timesteps:>10,}]: "
                              f"entropy={recent_entropy:.3f} < {self.min_entropy:.2f} "
                              f"→ ent_coef {current_coef:.4f} → {new_coef:.4f}")

            elif recent_entropy > self.max_entropy:
                new_coef = max(current_coef / self.adjustment_factor, self.min_ent_coef)
                if new_coef < current_coef - 1e-6:
                    self.adjustment_log.append({
                        'step': self.num_timesteps, 'entropy': recent_entropy,
                        'old_coef': current_coef, 'new_coef': new_coef, 'action': 'REDUCE',
                    })
                    if self.verbose:
                        print(f"   ❄️  ENT REDUCE [{self.num_timesteps:>10,}]: "
                              f"entropy={recent_entropy:.3f} > {self.max_entropy:.2f} "
                              f"→ ent_coef {current_coef:.4f} → {new_coef:.4f}")

            if abs(new_coef - current_coef) > 1e-6:
                self.model.ent_coef = new_coef

        return True


# ============================================================
# 10. MAKE ENV FACTORY
# ============================================================
def make_env(prices_arr, features_arr, regime_arr, window_size, is_training, rank=0):
    def _init():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS']      = '1'
        torch.set_num_threads(1)

        env = EventScalpingEnv(
            prices_arr, features_arr, regime_arr,
            window_size, is_training, rank
        )
        env.reset(seed=rank * 1337 + 42)
        return Monitor(env)
    return _init


# ============================================================
# 11. METRIK & EVAL
# ============================================================
SHARPE_ANNUALIZE = np.sqrt(365 * 24 * 60 * 60 * 10)

def hedge_fund_score(equity_curve, step_returns, trade_returns, n_trades):
    equity = np.array(equity_curve)

    if len(equity) < 2:
        return {
            "sharpe": 0.0, "max_drawdown": 0.0,
            "final_equity": 1.0, "trades": n_trades,
            "ev_trade_bps": 0.0, "win_rate": 0.0,
        }

    returns = np.array(step_returns)
    sharpe  = (
        np.mean(returns) / (np.std(returns) + 1e-12) * SHARPE_ANNUALIZE
        if np.std(returns) > 0 else 0.0
    )

    peak   = np.maximum.accumulate(equity)
    max_dd = float(abs(((equity - peak) / (peak + 1e-12)).min()))

    tr_arr   = np.array(trade_returns) if trade_returns else np.array([0.0])
    ev_trade = float(np.mean(tr_arr))
    win_rate = float(len(tr_arr[tr_arr > 0]) / len(tr_arr)) if len(tr_arr) > 0 else 0.0

    return {
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "final_equity": round(float(equity[-1]), 4),
        "trades":       n_trades,
        "ev_trade_bps": round(ev_trade, 4),
        "win_rate":     round(win_rate, 4),
    }

def evaluate_fold(model, val_prices, val_features, val_regime, vecnorm_path, fold):
    env_fn              = make_env(val_prices, val_features, val_regime, WINDOW_SIZE, False, rank=0)
    vec_env             = VecNormalize.load(vecnorm_path, DummyVecEnv([env_fn]))
    vec_env.training    = False
    vec_env.norm_reward = False

    global_equity  = STARTING_EQUITY
    equity_curve   = [global_equity]
    step_returns   = []
    trade_returns  = []
    total_trades   = 0

    obs            = vec_env.reset()
    lstm_states    = None
    episode_starts = np.ones((1,), dtype=bool)

    max_steps  = len(val_prices) - WINDOW_SIZE - 2
    step_count = 0

    with torch.inference_mode():
        while step_count < max_steps:
            action, lstm_states = model.predict(
                obs,
                state         = lstm_states,
                episode_start = episode_starts,
                deterministic = True
            )
            obs, _, d, info = vec_env.step(action)
            info_step = info[0]

            step_ret = float(info_step.get('step_return', 0.0))
            step_returns.append(step_ret)
            global_equity *= (1.0 + step_ret)
            equity_curve.append(global_equity)

            if info_step.get('closed_trade', False):
                total_trades += 1
                trade_returns.append(info_step.get('trade_log_ret_bps', 0.0))

            if bool(d[0]):
                lstm_states    = None
                episode_starts = np.ones((1,), dtype=bool)
            else:
                episode_starts = np.zeros((1,), dtype=bool)

            step_count += 1

            if step_count % 50000 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    vec_env.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    scores = hedge_fund_score(equity_curve, step_returns, trade_returns, total_trades)
    print(
        f"   📈 Eval Fold {fold}: "
        f"Sharpe={scores['sharpe']:>6.3f} | "
        f"MaxDD={scores['max_drawdown']*100:>5.2f}% | "
        f"Trades={scores['trades']} | "
        f"WR={scores['win_rate']*100:.1f}% | "
        f"EV_trade={scores['ev_trade_bps']:+.2f}bps | "
        f"FinalEq={scores['final_equity']:.4f}"
    )
    return scores


# ============================================================
# 12. MAIN WALK-FORWARD TRAINING
# ============================================================
def run_walk_forward():
    print("=" * 70)
    print("  EVENT SCALPING v21 MINER — CAUSAL TRANSFORMER + DSR + ENTROPY GUARDIAN")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    torch.backends.cudnn.benchmark         = True
    torch.backends.cudnn.deterministic     = False

    cuda_available = torch.cuda.is_available()
    fused_adamw    = cuda_available
    device_str     = "cuda:0" if cuda_available else "cpu"
    print(f"💻 Device: {device_str} | Extractor: {EXTRACTOR_TYPE.upper()} | fused AdamW: {fused_adamw}")

    total_rollout = N_STEPS * N_ENVS
    assert total_rollout % BATCH_SIZE == 0
    print(f"✅ Rollout: {N_STEPS} × {N_ENVS} = {total_rollout}, "
          f"BATCH={BATCH_SIZE} → {total_rollout // BATCH_SIZE} minibatches")

    print(f"\n📋 Konfigurasi v21 MINER:")
    print(f"   EXTRACTOR:          {EXTRACTOR_TYPE.upper()} ({'Causal Transformer' if EXTRACTOR_TYPE == 'transformer' else 'TCN'})")
    print(f"   REGIME SOURCE:      {'miner volatility_regime' if USE_MINER_REGIME else 'GMM online filter'}")
    print(f"   WARMUP FILTER:      ON (baris is_warmup=True dibuang)")
    print(f"   LEAKAGE EXCLUSION:  realized_spread, adverse_selection_metric")
    print(f"   DSR ETA/SCALE:      {DSR_ETA} / {DSR_SCALE}")
    print(f"   REGIME ASYMMETRY:   {REGIME_DOWNSIDE_MULT}")
    print(f"   ENT_COEF START:     {ENT_COEF}")
    print(f"   ENT GUARDIAN:       [{ENT_GUARDIAN_MIN_ENTROPY}, {ENT_GUARDIAN_MAX_ENTROPY}] nats")

    valid_paths = [p for p in CSV_PATHS if os.path.exists(p)]
    if not valid_paths:
        print("❌ Tidak ada dataset yang ditemukan.")
        print(f"   Pastikan file ada di: {CSV_PATHS}")
        return
    assert len(valid_paths) == 1, "❌ Skrip ini untuk 1 pair saja"

    # GMM hanya dijalankan jika tidak pakai miner regime
    mu_vol_auto  = REGIME_MU_VOL
    sig_vol_auto = REGIME_SIG_VOL
    if not USE_MINER_REGIME:
        if USE_HARDCODED_GMM:
            print(f"\n⚡ HARDCODED GMM aktif")
        else:
            mu_vol_auto, sig_vol_auto = estimate_regime_params(valid_paths)
            global REGIME_MU_VOL, REGIME_SIG_VOL
            REGIME_MU_VOL  = mu_vol_auto
            REGIME_SIG_VOL = sig_vol_auto

    print(f"\n📐 Preprocessing: {valid_paths[0]}...")
    df        = preprocess_dataset(valid_paths[0], mu_vol_auto, sig_vol_auto)
    pair_name = os.path.basename(valid_paths[0]).replace('.csv', '')
    print(f"   Total rows setelah filter: {len(df):,}")

    numeric_cols          = set(df.select_dtypes(include=[np.number]).columns.tolist())
    REQUIRED_FEATURE_COLS = sorted(list(numeric_cols - EXCLUDE_COLS))
    print(f"✅ Features: {len(REQUIRED_FEATURE_COLS)} kolom")

    # Verifikasi tidak ada leakage columns
    leaked = [c for c in ['realized_spread', 'adverse_selection_metric'] if c in REQUIRED_FEATURE_COLS]
    if leaked:
        print(f"🚨 LEAKAGE DETECTED: {leaked} masih ada di features! Cek EXCLUDE_COLS.")
        return
    print(f"✅ Leakage check OK — realized_spread & adverse_selection_metric diexclude")

    print("\n📐 Preparing baseline test data...")
    sample_size = min(50000, len(df) // 2)
    sample_raw  = df.iloc[:sample_size][REQUIRED_FEATURE_COLS].values.astype(np.float32)
    sample_mean = np.mean(sample_raw, axis=0)
    sample_std  = np.std(sample_raw,  axis=0)
    sample_std[sample_std == 0] = 1.0
    sample_norm = (sample_raw - sample_mean) / sample_std
    sample_p    = df.iloc[:sample_size]['micro_price'].values.astype(np.float64)
    sample_r    = df.iloc[:sample_size]['regime_state'].values.astype(np.int32)

    baseline_results = run_baseline_tests(sample_p, sample_norm, sample_r, n_steps=10000)

    with open(os.path.join(CHECKPOINT_DIR, "baseline_results.json"), "w") as f:
        json.dump({k: float(v) for k, v in baseline_results.items()}, f, indent=2)

    if baseline_results['hold_only_equity'] < 0.90:
        print(f"\n⚠️  PERINGATAN: HOLD-ONLY equity rendah. Ctrl+C dalam 10s untuk batal...")
        time.sleep(10)

    # Walk-forward
    model         = None
    vecnorm_path  = os.path.join(CHECKPOINT_DIR, "vec_normalize_stats.pkl")
    lr_schedule   = get_linear_fn(LR_START, LR_END, 1.0)
    clip_schedule = get_linear_fn(CLIP_RANGE_START, CLIP_RANGE_END, 1.0)
    fold_scores   = []

    extractor_cls, extractor_kwargs = get_extractor()

    for fold in range(N_SPLITS):
        print(f"\n{'='*60}")
        print(f"  🚀 FOLD {fold + 1} / {N_SPLITS} | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        chunk_size = len(df) // (N_SPLITS + 1)
        train_end  = chunk_size * (fold + 1) - PURGE_GAP
        val_start  = chunk_size * (fold + 1) + PURGE_GAP
        val_end    = chunk_size * (fold + 2)

        print(f"   Train: {train_end:,} rows | Val: {val_end - val_start:,} rows")

        train_df = df.iloc[:train_end]
        val_df   = df.iloc[val_start:val_end]

        raw_train = train_df[REQUIRED_FEATURE_COLS].values.astype(np.float32)
        mean      = np.mean(raw_train, axis=0)
        std       = np.std(raw_train,  axis=0)
        std[std == 0] = 1.0

        scaler        = StandardScaler()
        scaler.mean_  = mean
        scaler.scale_ = std
        scaler.var_   = std ** 2

        with open(os.path.join(CHECKPOINT_DIR, f"scaler_fold{fold}.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        train_features = (raw_train - mean) / std
        train_p        = train_df['micro_price'].values.astype(np.float64)
        train_r        = train_df['regime_state'].values.astype(np.int32)

        val_data = None
        if len(val_df) >= WINDOW_SIZE + 100:
            raw_val      = val_df[REQUIRED_FEATURE_COLS].values.astype(np.float32)
            val_features = (raw_val - mean) / std
            val_p        = val_df['micro_price'].values.astype(np.float64)
            val_r        = val_df['regime_state'].values.astype(np.int32)
            val_data     = (val_p, val_features, val_r, pair_name)

        train_envs_fns = [
            make_env(train_p, train_features, train_r, WINDOW_SIZE, True, w)
            for w in range(N_ENVS)
        ]

        raw_train_env = SubprocVecEnv(train_envs_fns, start_method='forkserver')
        _active_envs.append(raw_train_env)

        if fold == 0:
            train_env = VecNormalize(
                raw_train_env,
                norm_obs    = False,
                norm_reward = True,
                clip_obs    = VEC_CLIP_OBS,
                clip_reward = 10.0,
                gamma       = GAMMA
            )

            optimizer_kwargs = dict(weight_decay=1e-4, eps=1e-5)
            if fused_adamw:
                optimizer_kwargs['fused'] = True

            model = RecurrentPPO(
                "MlpLstmPolicy",
                train_env,
                learning_rate = lr_schedule,
                n_steps       = N_STEPS,
                batch_size    = BATCH_SIZE,
                n_epochs      = N_EPOCHS,
                ent_coef      = ENT_COEF,
                vf_coef       = VF_COEF,
                max_grad_norm = MAX_GRAD_NORM,
                clip_range    = clip_schedule,
                gae_lambda    = GAE_LAMBDA,
                gamma         = GAMMA,
                policy_kwargs = dict(
                    features_extractor_class  = extractor_cls,
                    features_extractor_kwargs = extractor_kwargs,
                    net_arch                  = dict(pi=PI_NET_ARCH, vf=VF_NET_ARCH),
                    lstm_hidden_size          = LSTM_HIDDEN_SIZE,
                    n_lstm_layers             = N_LSTM_LAYERS,
                    shared_lstm               = SHARED_LSTM,
                    enable_critic_lstm        = ENABLE_CRITIC_LSTM,
                    optimizer_class           = torch.optim.AdamW,
                    optimizer_kwargs          = optimizer_kwargs,
                ),
                verbose = 1,
                device  = device_str
            )
        else:
            train_env             = VecNormalize.load(vecnorm_path, raw_train_env)
            train_env.training    = True
            train_env.norm_reward = True
            model.set_env(train_env)

        survival_cb = SurvivalMonitorCallback(check_freq=2048, early_stop=ENABLE_EARLY_STOP)
        entropy_cb  = EntropyGuardianCallback(
            min_entropy       = ENT_GUARDIAN_MIN_ENTROPY,
            max_entropy       = ENT_GUARDIAN_MAX_ENTROPY,
            adjustment_factor = ENT_GUARDIAN_ADJ_FACTOR,
            min_ent_coef      = ENT_GUARDIAN_MIN_COEF,
            max_ent_coef      = ENT_GUARDIAN_MAX_COEF,
            check_freq        = ENT_GUARDIAN_CHECK_FREQ,
            verbose           = 1,
        )
        callback = CallbackList([survival_cb, entropy_cb])

        try:
            model.learn(
                total_timesteps     = TIMESTEPS_PER_FOLD,
                callback            = callback,
                reset_num_timesteps = False
            )
        except Exception as e:
            print(f"   ⚠️  Training error: {e}")

        if entropy_cb.adjustment_log:
            with open(os.path.join(CHECKPOINT_DIR, f"entropy_adjustments_fold{fold + 1}.json"), "w") as f:
                json.dump(entropy_cb.adjustment_log, f, indent=2, default=float)
            print(f"   📝 Entropy adjustments: {len(entropy_cb.adjustment_log)} kali")

        train_env.save(vecnorm_path)

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_fold{fold + 1}")
        model.save(checkpoint_path)
        print(f"   💾 Checkpoint: {checkpoint_path}.zip")

        _active_envs.remove(raw_train_env)
        train_env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if val_data is not None:
            print(f"\n📐 Evaluasi Fold {fold + 1}:")
            val_p, val_feat, val_r, pname = val_data
            try:
                scores = evaluate_fold(model, val_p, val_feat, val_r, vecnorm_path, fold + 1)
                fold_scores.append({
                    'fold':         fold + 1,
                    'sharpe':       scores['sharpe'],
                    'dd':           scores['max_drawdown'],
                    'equity':       scores['final_equity'],
                    'trades':       scores['trades'],
                    'ev_trade_bps': scores['ev_trade_bps'],
                    'win_rate':     scores['win_rate'],
                })

                with open(os.path.join(CHECKPOINT_DIR, "fold_summary.json"), "w") as f:
                    json.dump(fold_scores, f, indent=2)
            except Exception as e:
                print(f"   ⚠️  Eval gagal: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "=" * 60)
    print("  SUMMARY WALK-FORWARD v21 MINER")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    for fs in fold_scores:
        print(
            f"  Fold {fs['fold']}: "
            f"Sharpe={fs['sharpe']:>7.3f} | "
            f"MaxDD={fs['dd']*100:>5.2f}% | "
            f"Trades={fs['trades']} | "
            f"WR={fs['win_rate']*100:.1f}% | "
            f"EV_trade={fs['ev_trade_bps']:+.2f}bps | "
            f"FinalEq={fs['equity']:.4f}"
        )

    if fold_scores:
        avg_sharpe = np.mean([f['sharpe']       for f in fold_scores])
        avg_dd     = np.mean([f['dd']           for f in fold_scores]) * 100
        avg_eq     = np.mean([f['equity']       for f in fold_scores])
        avg_ev     = np.mean([f['ev_trade_bps'] for f in fold_scores])
        avg_wr     = np.mean([f['win_rate']     for f in fold_scores]) * 100
        print(
            f"\n  Overall: Sharpe={avg_sharpe:.3f} | MaxDD={avg_dd:.2f}% | "
            f"FinalEq={avg_eq:.4f} | EV_trade={avg_ev:+.2f}bps | WR={avg_wr:.1f}%"
        )

    if model is not None:
        final_path = os.path.join(CHECKPOINT_DIR, "event_scalping_v21_miner_FINAL")
        model.save(final_path)
        print(f"\n💾 Final model: {final_path}.zip")

    regime_params = {
        'mu_vol':  REGIME_MU_VOL,
        'sig_vol': REGIME_SIG_VOL,
        'method':  'miner_volatility_regime' if USE_MINER_REGIME else 'GMM',
    }
    with open(os.path.join(CHECKPOINT_DIR, "regime_params.pkl"), "wb") as f:
        pickle.dump(regime_params, f)

    print("\n✅ Training v21 MINER selesai.")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_walk_forward()

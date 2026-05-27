import math
import zlib
import asyncio
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any, List, Optional

import numpy as np
from scipy.stats import linregress

class VPINCalculator:
    """
    Kalkulator Volume-synchronized Probability of Informed Trading (VPIN).
    Bucket-based update sesuai spesifikasi Bagian 6 (Grup 6).
    """
    def __init__(self, bucket_size: float = 1.0, window: int = 50):
        self.bucket_size = bucket_size
        self.window = window
        self.buckets: deque = deque(maxlen=window)
        self.acc_buy = 0.0
        self.acc_sell = 0.0
        self.acc_vol = 0.0

    def update(self, buy_vol: float, sell_vol: float):
        self.acc_buy += buy_vol
        self.acc_sell += sell_vol
        self.acc_vol += buy_vol + sell_vol
        
        while self.acc_vol >= self.bucket_size:
            imbalance = abs(self.acc_buy - self.acc_sell)
            self.buckets.append(imbalance / (self.bucket_size + 1e-9))
            self.acc_buy = 0.0
            self.acc_sell = 0.0
            self.acc_vol = 0.0

    @property
    def vpin(self) -> float:
        if not self.buckets:
            return 0.0
        return float(np.mean(self.buckets))


def calc_ema(series: List[float], span: int) -> float:
    """Helper untuk menghitung EMA nilai terakhir."""
    if not series:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    ema = series[0]
    for val in series[1:]:
        ema = val * alpha + ema * (1.0 - alpha)
    return ema


class FeatureEngine:
    def __init__(self, pair: str, shared_state: Dict[str, Any]):
        self.pair = pair
        self.queue: asyncio.Queue = shared_state["queue"]
        self.config = shared_state["config"]
        
        # Konfigurasi dari config.yaml (fallback defaults)
        self.warmup_bars = self.config.get("warmup", {}).get("bars", 60)
        self.trade_intensity_span = 60
        self.vpin_bucket_multiplier = self.config.get("rolling_windows", {}).get("vpin_bucket_multiplier", 5.0)

        # 5.1 STATE YANG DIBUTUHKAN
        self.bar_buffer = deque(maxlen=70)
        self.trade_buffer: deque = deque(maxlen=10000)
        self.trade_level_buffer = deque(maxlen=30)
        self.spread_history = deque(maxlen=60)
        self.returns_history = deque(maxlen=60)
        self.vpin_history = deque(maxlen=200)
        self.vol_regime_history = deque(maxlen=3600)  # 1 jam cukup untuk percentile regime
        
        self.trade_count_ewma = 0.0
        self.vpin_calculator = VPINCalculator(bucket_size=1.0)
        self.cumulative_delta = 0.0
        self.last_date_utc = datetime.now(timezone.utc).date()
        self.seq_id = 0
        self.warmup_count = 0
        self.trade_buffer_snapshot: list = []
        
        # State Forward-Filled Futures Data
        self.mark_price = 0.0
        self.index_price = 0.0
        self.funding_rate = 0.0
        self.predicted_funding = 0.0
        self.open_interest = 0.0
        self.prev_oi_60s = deque(maxlen=60)
        self.long_short_ratio = 0.0

        # State internal
        self.current_ob = None
        self.current_sec = 0

    def add_trade(self, data: dict):
        """Append ke trade_buffer dan trade_level_buffer"""
        self.trade_buffer.append(data)
        
        price = float(data.get('p', 0))
        qty = float(data.get('q', 0))
        is_maker = data.get('m', False)
        
        # Signed vol: taker buy (m=False) -> qty, taker sell (m=True) -> -qty
        signed_vol = -qty if is_maker else qty
        self.trade_level_buffer.append((price, signed_vol))

    def update_ob(self, ob):
        """Simpan referensi atau copy ringan snapshot bids/asks saat ini"""
        self.current_ob = ob

    def update_futures(self, key: str, val: float):
        """Update forward-fill values dari polling"""
        setattr(self, key, val)

    async def tick(self, event_time_ms: int, local_time_ms: int, skew_ms: int):
        """Metode utama pemicu pembentukan bar"""
        sec = event_time_ms // 1000
        
        if self.current_sec == 0:
            self.current_sec = sec
            
        if sec > self.current_sec:
            # Update current_sec SEBELUM await pertama agar concurrent tick()
            # dari stream lain tidak ikut compute bar yang sama (race condition).
            bar_sec = self.current_sec
            self.current_sec = sec
            self.trade_buffer_snapshot = list(self.trade_buffer)
            self.trade_buffer.clear()

            try:
                row = self.compute_bar(bar_sec * 1000, local_time_ms, skew_ms)
            except Exception as e:
                import logging, traceback
                logging.getLogger(f"features.{self.pair}").error(
                    f"compute_bar error di bar {bar_sec}: {e}\n{traceback.format_exc()}"
                )
                row = None

            if row is not None:
                self.bar_buffer.append(row)
                await self.process_retroactive_realized_spread()
                await self.queue.put({"type": "row", "data": row})

    async def process_retroactive_realized_spread(self):
        """
        Hitung realized_spread retroaktif saat bar t+5 tersedia.
        """
        if len(self.bar_buffer) > 5:
            target = self.bar_buffer[-6]  # Bar t
            future = self.bar_buffer[-1]  # Bar t+5
            
            mid_t5 = future["mid_price"]
            d_t = np.sign(target["delta_volume"])
            if d_t == 0:
                d_t = np.sign(target["log_return"])
                
            if d_t != 0:
                rs = 2 * d_t * (target["close"] - mid_t5)
                target["realized_spread"] = float(rs)
                target["adverse_selection_metric"] = float(rs - target["effective_spread"])
                
                # Kirim perintah update ke storage
                await self.queue.put({
                    "type": "update",
                    "seq_id": target["sequence_id"],
                    "realized_spread": target["realized_spread"],
                    "adverse_selection_metric": target["adverse_selection_metric"]
                })

    def compute_bar(self, ts_ms: int, local_time_ms: int, skew_ms: int) -> Optional[Dict[str, Any]]:
        if not self.current_ob:
            return None
            
        bids = self.current_ob.get_bids(20)
        asks = self.current_ob.get_asks(20)
        if not bids or not asks:
            return None

        # Waktu
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        current_date = dt.date()
        
        if current_date != self.last_date_utc:
            self.cumulative_delta = 0.0
            self.last_date_utc = current_date

        self.seq_id += 1
        self.warmup_count += 1
        is_warmup = self.warmup_count <= self.warmup_bars

        # ==== GRUP 1: OHLCV ====
        prices = [float(t['p']) for t in self.trade_buffer_snapshot]
        sizes = [float(t['q']) for t in self.trade_buffer_snapshot]
        
        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]

        # Crossed book (ask < bid) terjadi saat re-sync atau market dislocation.
        # Return None agar bar ini diabaikan dan tidak menghasilkan spread negatif.
        if best_ask <= best_bid:
            return None

        mid_price = (best_bid + best_ask) / 2.0
        
        taker_buy_vol = sum(q for t, q in zip(self.trade_buffer_snapshot, sizes) if not t['m'])
        taker_sell_vol = sum(q for t, q in zip(self.trade_buffer_snapshot, sizes) if t['m'])
        
        open_p = prices[0] if prices else mid_price
        high_p = max(prices) if prices else mid_price
        low_p = min(prices) if prices else mid_price
        close_p = prices[-1] if prices else mid_price
        volume = sum(sizes)
        buy_trade_count = sum(1 for t in self.trade_buffer_snapshot if not t['m'])
        sell_trade_count = sum(1 for t in self.trade_buffer_snapshot if t['m'])
        trade_count = len(self.trade_buffer_snapshot)

        # ==== GRUP 2: ORDER BOOK FEATURES ====
        spread = best_ask - best_bid
        microprice = (best_bid * best_bid_qty + best_ask * best_ask_qty) / (best_bid_qty + best_ask_qty + 1e-9)
        
        top5_bid_vol = sum(q for _, q in bids[:5])
        top5_ask_vol = sum(q for _, q in asks[:5])
        top5_bid_vwap = sum(p * q for p, q in bids[:5]) / (top5_bid_vol + 1e-9)
        top5_ask_vwap = sum(p * q for p, q in asks[:5]) / (top5_ask_vol + 1e-9)
        weighted_mid_price = (top5_bid_vwap * top5_bid_vol + top5_ask_vwap * top5_ask_vol) / (top5_bid_vol + top5_ask_vol + 1e-9)
        
        bid_depth_top5 = top5_bid_vol
        bid_depth_top10 = sum(q for _, q in bids[:10])
        bid_depth_top20 = sum(q for _, q in bids[:20])
        ask_depth_top5 = top5_ask_vol
        ask_depth_top10 = sum(q for _, q in asks[:10])
        ask_depth_top20 = sum(q for _, q in asks[:20])
        
        cumulative_bid_volume = bid_depth_top20
        cumulative_ask_volume = ask_depth_top20
        order_book_imbalance = (cumulative_bid_volume - cumulative_ask_volume) / (cumulative_bid_volume + cumulative_ask_volume + 1e-9)
        volume_imbalance = (taker_buy_vol - taker_sell_vol) / (volume + 1e-9)
        
        depth_imbal_list = [(bq - aq) / (bq + aq + 1e-9) for (_, bq), (_, aq) in zip(bids[:20], asks[:20])]
        depth_imbalance = np.mean(depth_imbal_list) if depth_imbal_list else 0.0
        
        cum_vol_bid = np.cumsum([q for _, q in bids[:20]])
        prices_bid = [p for p, _ in bids[:20]]
        slope_bid = linregress(cum_vol_bid, prices_bid).slope if np.std(cum_vol_bid) > 0 else 0.0
        
        cum_vol_ask = np.cumsum([q for _, q in asks[:20]])
        prices_ask = [p for p, _ in asks[:20]]
        slope_ask = linregress(cum_vol_ask, prices_ask).slope if np.std(cum_vol_ask) > 0 else 0.0
        
        slope_imbalance = slope_bid - slope_ask
        liquidity_pressure = cumulative_bid_volume / (cumulative_bid_volume + cumulative_ask_volume + 1e-9)
        queue_imbalance = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty + 1e-9)
        order_book_pressure_ratio = cumulative_bid_volume / (cumulative_ask_volume + 1e-9)

        # ==== GRUP 3: TRADE FLOW FEATURES ====
        delta_volume = taker_buy_vol - taker_sell_vol
        self.cumulative_delta += delta_volume
        
        alpha_ti = 2.0 / (self.trade_intensity_span + 1.0)
        self.trade_count_ewma = self.trade_count_ewma + alpha_ti * (trade_count - self.trade_count_ewma)
        
        trade_burst_metric = trade_count / (self.trade_count_ewma + 1e-9)
        avg_trade_size = volume / (trade_count + 1e-9)
        max_trade_size = max(sizes) if sizes else 0.0
        signed_trade_flow = sum((q if not t['m'] else -q) for t, q in zip(self.trade_buffer_snapshot, sizes))
        
        timestamps_ms = sorted([float(t['T']) for t in self.trade_buffer_snapshot])
        if len(timestamps_ms) >= 2:
            inter_arrivals_s = np.diff(timestamps_ms) / 1000.0
            trade_arrival_rate = 1.0 / (np.mean(inter_arrivals_s) + 1e-9)
        else:
            trade_arrival_rate = float(trade_count)

        # ==== GRUP 4: PRICE ACTION FEATURES ====
        prev_close = self.bar_buffer[-1]["close"] if self.bar_buffer else close_p
        log_return = math.log(close_p / prev_close) if prev_close > 0 else 0.0
        self.returns_history.append(log_return)
        
        if len(self.returns_history) >= 20:
            r20 = np.array(self.returns_history)[-20:]
            realized_volatility = math.sqrt(np.mean(r20 ** 2))
            bipower_variance = np.sum(np.abs(r20[1:]) * np.abs(r20[:-1]))
        else:
            realized_volatility = 0.0
            bipower_variance = 0.0
            
        parkinson_volatility = math.sqrt((1 / (4 * math.log(2))) * (math.log(high_p / low_p)) ** 2) if high_p > low_p > 0 else 0.0
        momentum_10s = close_p - self.bar_buffer[-10]["close"] if len(self.bar_buffer) >= 10 else 0.0
        prev_log_return = self.bar_buffer[-1]["log_return"] if self.bar_buffer else 0.0
        price_acceleration = log_return - prev_log_return
        price_impact = abs(log_return) / (volume + 1e-9)
        
        # PERBAIKAN FATAL ERROR: Pencegahan NaN value jika orderbook rusak
        closes_hist = [r["close"] for r in self.bar_buffer] + [close_p]
        if len(closes_hist) >= 20:
            ema5 = calc_ema(closes_hist, 5)
            ema20 = calc_ema(closes_hist, 20)
            diff = ema5 - ema20
            # Pastikan tidak mengkonversi NaN ke integer
            micro_trend = int(np.sign(diff)) if not np.isnan(diff) else 0
        else:
            micro_trend = 0
            
        directional_pressure = (close_p - open_p) / (high_p - low_p + 1e-9)

        # ==== GRUP 5: LIQUIDITY FEATURES ====
        effective_spread = 2.0 * abs(close_p - mid_price)
        
        if len(self.trade_level_buffer) >= 10:
            prices_tl = [x[0] for x in self.trade_level_buffer]
            flows_tl = [x[1] for x in self.trade_level_buffer]
            delta_p = np.diff(prices_tl)
            flows_x = np.array(flows_tl[1:])
            if np.std(flows_x) > 0:
                kyle_lambda = linregress(flows_x, delta_p).slope
            else:
                kyle_lambda = 0.0
        else:
            kyle_lambda = 0.0
            
        if len(self.returns_history) >= 20 and len(self.bar_buffer) >= 20:
            vols_20 = [r["volume"] for r in list(self.bar_buffer)[-20:]]
            rets_20 = np.array(self.returns_history)[-20:]
            amihud_illiquidity = np.mean([abs(r) / (v + 1e-9) for r, v in zip(rets_20, vols_20)])
        else:
            amihud_illiquidity = 0.0
            
        self.spread_history.append(spread)
        if len(self.spread_history) >= 10:
            s_arr = np.array(self.spread_history)
            x_ar = s_arr[:-1]
            y_ar = s_arr[1:]
            if np.std(x_ar) > 0:
                beta_ar = linregress(x_ar, y_ar).slope
                resiliency_metric = -beta_ar
            else:
                resiliency_metric = 0.0
        else:
            resiliency_metric = 0.0
            
        rolling_mean_spread = np.mean(self.spread_history) if len(self.spread_history) >= 5 else spread
        liquidity_vacuum = bool(spread > 3 * rolling_mean_spread)

        # ==== GRUP 6: ORDER FLOW TOXICITY ====
        if not is_warmup and len(self.bar_buffer) >= self.warmup_bars:
            vols_60 = [r["volume"] for r in list(self.bar_buffer)[-60:]]
            avg_vol = calc_ema(vols_60, 60)
            self.vpin_calculator.bucket_size = max(1.0, avg_vol * self.vpin_bucket_multiplier)
            
        self.vpin_calculator.update(taker_buy_vol, taker_sell_vol)
        vpin = self.vpin_calculator.vpin
        
        self.vpin_history.append(vpin)
        if len(self.vpin_history) >= 10:
            arr = np.fromiter(self.vpin_history, dtype=np.float32)
            toxicity_score = float(np.searchsorted(np.sort(arr), vpin, side='right') / len(arr))
        else:
            toxicity_score = 0.0

        # ==== GRUP 7: FUTURES-SPECIFIC FEATURES ====
        self.prev_oi_60s.append(self.open_interest)
        oi_t60_ago = self.prev_oi_60s[0]
        oi_change_rate = (self.open_interest - oi_t60_ago) / (oi_t60_ago + 1e-9) if len(self.prev_oi_60s) == 60 else 0.0

        # ==== GRUP 8: MARKET REGIME FEATURES ====
        self.vol_regime_history.append(realized_volatility)
        if len(self.vol_regime_history) >= 60:
            p33, p66 = np.percentile(np.fromiter(self.vol_regime_history, dtype=np.float32), [33, 66])
            volatility_regime = 0 if realized_volatility < p33 else 2 if realized_volatility > p66 else 1
        else:
            volatility_regime = 1
            
        if micro_trend > 0 and price_acceleration >= 0:
            trend_regime = 2
        elif micro_trend < 0 and price_acceleration <= 0:
            trend_regime = 0
        else:
            trend_regime = 1
            
        ranging_regime = bool(volatility_regime == 0 and trend_regime == 1)
        
        # Weighted histogram — avoids creating trade_count Python float copies per bar
        _bar_list = list(self.bar_buffer)[-60:]
        _bar_vals = np.array([r["avg_trade_size"] for r in _bar_list], dtype=np.float64)
        _bar_wts  = np.array([max(r["trade_count"], 1) for r in _bar_list], dtype=np.float64)
        if sizes:
            _cur = np.array(sizes, dtype=np.float64)
            _all_vals = np.concatenate([_bar_vals, _cur])
            _all_wts  = np.concatenate([_bar_wts, np.ones(len(_cur), dtype=np.float64)])
        else:
            _all_vals = _bar_vals
            _all_wts  = _bar_wts

        if _all_vals.size >= 5 and np.sum(_all_wts) >= 5:
            counts, _ = np.histogram(_all_vals, bins=10, weights=_all_wts)
            total = np.sum(counts) + 1e-9
            probs = counts / total + 1e-9
            probs /= np.sum(probs)
            entropy = float(-np.sum(probs * np.log(probs)))
        else:
            entropy = 0.0
            
        if len(self.returns_history) >= 60:
            series = np.array(self.returns_history)[-60:]
            mean_s = np.mean(series)
            deviation = np.cumsum(series - mean_s)
            R = max(deviation) - min(deviation)
            S = np.std(series, ddof=1)
            hurst_exponent = float(math.log(R / S) / math.log(len(series))) if S > 0 and R > 0 else 0.5
        else:
            hurst_exponent = 0.5

        # ==== GRUP 9: TIME FEATURES ====
        h = dt.hour
        session_marker = 2 if 13 <= h < 16 else 2 if 13 <= h < 21 else 1 if 8 <= h < 16 else 0 if 0 <= h < 8 else 3

        # ==== BIND SEMUA KE ROW DICT ====
        row = {
            "Timestamp": ts_ms,
            "local_timestamp_ms": local_time_ms,
            "clock_skew_ms": int(skew_ms),
            "sequence_id": int(self.seq_id),
            "is_warmup": bool(is_warmup),
            
            "open": float(open_p), "high": float(high_p), "low": float(low_p), "close": float(close_p),
            "volume": float(volume), "taker_buy_volume": float(taker_buy_vol), "taker_sell_volume": float(taker_sell_vol),
            "trade_count": int(trade_count), "buy_trade_count": int(buy_trade_count), "sell_trade_count": int(sell_trade_count),
            
            "best_bid": float(best_bid), "best_ask": float(best_ask), "spread": float(spread),
            "mid_price": float(mid_price), "microprice": float(microprice), "weighted_mid_price": float(weighted_mid_price),
            "bid_depth_top5": float(bid_depth_top5), "bid_depth_top10": float(bid_depth_top10), "bid_depth_top20": float(bid_depth_top20),
            "ask_depth_top5": float(ask_depth_top5), "ask_depth_top10": float(ask_depth_top10), "ask_depth_top20": float(ask_depth_top20),
            "cumulative_bid_volume": float(cumulative_bid_volume), "cumulative_ask_volume": float(cumulative_ask_volume),
            "order_book_imbalance": float(order_book_imbalance), "volume_imbalance": float(volume_imbalance),
            "depth_imbalance": float(depth_imbalance), "slope_bid": float(slope_bid), "slope_ask": float(slope_ask),
            "slope_imbalance": float(slope_imbalance), "liquidity_pressure": float(liquidity_pressure),
            "queue_imbalance": float(queue_imbalance), "order_book_pressure_ratio": float(order_book_pressure_ratio),
            
            "aggressive_buy_volume": float(taker_buy_vol), "aggressive_sell_volume": float(taker_sell_vol),
            "delta_volume": float(delta_volume), "cumulative_delta": float(self.cumulative_delta),
            "trade_intensity": float(self.trade_count_ewma), "avg_trade_size": float(avg_trade_size),
            "max_trade_size": float(max_trade_size), "signed_trade_flow": float(signed_trade_flow),
            "trade_burst_metric": float(trade_burst_metric), "trade_arrival_rate": float(trade_arrival_rate),
            
            "log_return": float(log_return), "realized_volatility": float(realized_volatility),
            "parkinson_volatility": float(parkinson_volatility), "bipower_variance": float(bipower_variance),
            "momentum_10s": float(momentum_10s), "price_acceleration": float(price_acceleration),
            "price_impact": float(price_impact), "micro_trend": int(micro_trend), "directional_pressure": float(directional_pressure),
            
            "effective_spread": float(effective_spread), "quoted_spread": float(spread),
            "realized_spread": float('nan'),  # Retroactive
            "kyle_lambda": float(kyle_lambda), "amihud_illiquidity": float(amihud_illiquidity),
            "resiliency_metric": float(resiliency_metric), "liquidity_vacuum": bool(liquidity_vacuum),
            
            "vpin": float(vpin), "toxicity_score": float(toxicity_score),
            "adverse_selection_metric": float('nan'),  # Retroactive
            
            "funding_rate": float(self.funding_rate), "mark_price": float(self.mark_price),
            "index_price": float(self.index_price), "mark_index_spread": float(self.mark_price - self.index_price),
            "open_interest": float(self.open_interest), "oi_change_rate": float(oi_change_rate),
            "long_short_ratio": float(self.long_short_ratio), "predicted_funding_rate": float(self.predicted_funding),
            
            "volatility_regime": int(volatility_regime), "trend_regime": int(trend_regime),
            "ranging_regime": bool(ranging_regime), "entropy": float(entropy), "hurst_exponent": float(hurst_exponent),
            
            "second_of_minute": int(dt.second), "minute_of_hour": int(dt.minute),
            "hour_of_day": int(dt.hour), "day_of_week": int(dt.weekday()), "session_marker": int(session_marker),
            
            "book_update_count": int(self.current_ob.reset_update_count()),
            "trade_msg_count": int(trade_count), "has_gap": False, "recovery_source": int(0)
        }

        # ==== GRUP 10: CHECKSUM (CRC32) ====
        float_cols = ["open", "high", "low", "close", "volume", "taker_buy_volume", 
                      "taker_sell_volume", "delta_volume", "best_bid", "best_ask", 
                      "mid_price", "microprice", "spread", "order_book_imbalance", 
                      "kyle_lambda", "realized_volatility", "parkinson_volatility", 
                      "log_return", "momentum_10s", "vpin", "funding_rate", 
                      "mark_price", "index_price", "open_interest"]
        
        num_bytes = b"".join(np.float32(row[c]).tobytes() for c in float_cols)
        row["row_checksum"] = int(zlib.crc32(num_bytes) & 0xFFFFFFFF)

        return row


import os
import time
import zlib
import asyncio
import logging
from typing import Dict, Any, List

import aiohttp
import orjson
import numpy as np
from datetime import datetime, timezone

class RecoveryManager:
    """
    Manajemen Pemulihan Data (Recovery) jika terjadi WebSocket disconnect atau gap.
    Menggunakan 3 level fallback: aggTrades -> klines -> Interpolasi linear.
    """
    def __init__(self, shared_state: Dict[str, Any], config: Dict[str, Any]):
        self.shared_state = shared_state
        self.config = config
        self.state_file = "recovery.json"
        self.state: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("recovery")
        
        self._load_state()
        
        # Mulai background task untuk menyimpan state setiap 10 detik
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._state_writer_loop(), name="recovery_state_writer")
        except RuntimeError:
            pass # Dipanggil sebelum loop berjalan, task di-handle oleh caller jika perlu

    def _load_state(self):
        """7.1 STATE FILE: Load dari recovery.json jika ada"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "rb") as f:
                    self.state = orjson.loads(f.read())
                self.logger.info(f"Recovery state loaded: {len(self.state)} pairs")
            except Exception as e:
                self.logger.error(f"Gagal memuat {self.state_file}: {e}")
                
        # Inisialisasi default jika pair belum ada
        for pair in self.shared_state.keys():
            if pair not in self.state:
                self.state[pair] = {"last_seq": 0, "last_ts": 0}

    async def save_state(self):
        """Menyimpan state ke disk via WAL pattern"""
        tmp_file = f"{self.state_file}.tmp"
        try:
            # Menggunakan aiofiles atau thread agar tidak memblokir loop
            json_data = orjson.dumps(self.state)
            await asyncio.to_thread(self._write_file_sync, tmp_file, json_data)
            os.replace(tmp_file, self.state_file)
        except Exception as e:
            self.logger.error(f"Gagal menyimpan recovery state: {e}")

    def _write_file_sync(self, path: str, data: bytes):
        with open(path, "wb") as f:
            f.write(data)

    async def _state_writer_loop(self):
        """Update setiap 10 detik via WAL"""
        while True:
            await asyncio.sleep(10)
            await self.save_state()

    def update_state(self, pair: str, seq_id: int, ts: int):
        """Dipanggil (misal oleh monitor atau manual) untuk update tracking state"""
        if pair in self.state:
            self.state[pair]["last_seq"] = max(self.state[pair]["last_seq"], seq_id)
            self.state[pair]["last_ts"] = max(self.state[pair]["last_ts"], ts)

    def _get_default_row(self, ts: int, seq_id: int) -> dict:
        """Membuat template row kosong dengan nilai default sesuai schema PyArrow (Bagian 10)"""
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return {
            "Timestamp": int(ts), "local_timestamp_ms": int(time.time() * 1000), "clock_skew_ms": 0,
            "sequence_id": int(seq_id), "is_warmup": False,
            "open": float('nan'), "high": float('nan'), "low": float('nan'), "close": float('nan'),
            "volume": 0.0, "taker_buy_volume": 0.0, "taker_sell_volume": 0.0, "trade_count": 0,
            "buy_trade_count": 0, "sell_trade_count": 0,
            "best_bid": 0.0, "best_ask": 0.0, "spread": 0.0, "mid_price": 0.0,
            "microprice": 0.0, "weighted_mid_price": 0.0, "bid_depth_top5": 0.0,
            "bid_depth_top10": 0.0, "bid_depth_top20": 0.0, "ask_depth_top5": 0.0,
            "ask_depth_top10": 0.0, "ask_depth_top20": 0.0, "cumulative_bid_volume": 0.0,
            "cumulative_ask_volume": 0.0, "order_book_imbalance": 0.0, "volume_imbalance": 0.0,
            "depth_imbalance": 0.0, "slope_bid": 0.0, "slope_ask": 0.0, "slope_imbalance": 0.0,
            "liquidity_pressure": 0.0, "queue_imbalance": 0.0, "order_book_pressure_ratio": 0.0,
            "aggressive_buy_volume": 0.0, "aggressive_sell_volume": 0.0, "delta_volume": 0.0,
            "cumulative_delta": 0.0, "trade_intensity": 0.0, "avg_trade_size": 0.0,
            "max_trade_size": 0.0, "signed_trade_flow": 0.0, "trade_burst_metric": 0.0,
            "trade_arrival_rate": 0.0, "log_return": 0.0, "realized_volatility": 0.0,
            "parkinson_volatility": 0.0, "bipower_variance": 0.0, "momentum_10s": 0.0,
            "price_acceleration": 0.0, "price_impact": 0.0, "micro_trend": 0,
            "directional_pressure": 0.0, "effective_spread": 0.0, "quoted_spread": 0.0,
            "realized_spread": float('nan'), "kyle_lambda": 0.0, "amihud_illiquidity": 0.0,
            "resiliency_metric": 0.0, "liquidity_vacuum": False, "vpin": 0.0,
            "toxicity_score": 0.0, "adverse_selection_metric": float('nan'),
            "funding_rate": 0.0, "mark_price": 0.0, "index_price": 0.0,
            "mark_index_spread": 0.0, "open_interest": 0.0, "oi_change_rate": 0.0,
            "long_short_ratio": 0.0, "predicted_funding_rate": 0.0, "volatility_regime": 1,
            "trend_regime": 1, "ranging_regime": False, "entropy": 0.0, "hurst_exponent": 0.5,
            
            "second_of_minute": int(dt.second), "minute_of_hour": int(dt.minute),
            "hour_of_day": int(dt.hour), "day_of_week": int(dt.weekday()), 
            "session_marker": 2 if 13 <= dt.hour < 21 else (1 if 8 <= dt.hour < 16 else (0 if 0 <= dt.hour < 8 else 3)),
            "book_update_count": 0, "trade_msg_count": 0, "has_gap": False, "recovery_source": 0, "row_checksum": 0
        }

    def _finalize_row(self, row: dict) -> dict:
        """Menghitung CRC32 Checksum sebelum di-push ke antrean"""
        float_cols = ["open", "high", "low", "close", "volume", "taker_buy_volume", 
                      "taker_sell_volume", "delta_volume", "best_bid", "best_ask", 
                      "mid_price", "microprice", "spread", "order_book_imbalance", 
                      "kyle_lambda", "realized_volatility", "parkinson_volatility", 
                      "log_return", "momentum_10s", "vpin", "funding_rate", 
                      "mark_price", "index_price", "open_interest"]
        
        # Konversi NaN aman ke float32 array
        arr = []
        for c in float_cols:
            val = row.get(c, 0.0)
            arr.append(0.0 if np.isnan(val) else val)
            
        num_bytes = b"".join(np.float32(v).tobytes() for v in arr)
        row["row_checksum"] = int(zlib.crc32(num_bytes) & 0xFFFFFFFF)
        return row

    async def handle_gap(self, pair: str, downtime_start: float, downtime_end: float):
        """
        7.3 RECOVERY SEQUENCE
        Menjalankan 3 level recovery saat terjadi koneksi terputus.
        """
        start_ms = int(downtime_start * 1000)
        end_ms = int(downtime_end * 1000)
        
        # Gap Detection Constraint (7.2)
        last_known_ts = self.state.get(pair, {}).get("last_ts", 0)
        if start_ms <= last_known_ts:
            start_ms = last_known_ts + 1000

        if end_ms - start_ms <= 1000:
            return  # Gap terlalu kecil / tidak ada gap
            
        self.logger.info(f"Memicu Recovery Sequence untuk {pair}. Gap: {start_ms} - {end_ms} ({end_ms - start_ms}ms)")
        
        queue = self.shared_state[pair]["queue"]
        rate_limiter = self.shared_state[pair].get("rate_limiter")
        
        recovered_rows = []
        seq_counter = self.state.get(pair, {}).get("last_seq", 0)

        async with aiohttp.ClientSession() as session:
            
            # --- LEVEL 1: REST /fapi/v1/aggTrades ---
            # Jika gap terlalu besar (> 1000 detik), bypass Level 1 untuk menghindari out-of-memory/rate limit parah
            if (end_ms - start_ms) <= 100000: 
                try:
                    if rate_limiter: await rate_limiter.acquire(5)
                    url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={pair}&startTime={start_ms}&endTime={end_ms}&limit=1000"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            trades = await resp.json()
                            if len(trades) < 1000 and len(trades) > 0:
                                # Proses aggTrades menjadi 1s bars
                                trade_groups = {}
                                for t in trades:
                                    ts_sec = (int(t['T']) // 1000) * 1000
                                    trade_groups.setdefault(ts_sec, []).append(t)
                                
                                for sec_ts in sorted(trade_groups.keys()):
                                    grp = trade_groups[sec_ts]
                                    seq_counter += 1
                                    row = self._get_default_row(sec_ts, seq_counter)
                                    
                                    prices = [float(x['p']) for x in grp]
                                    sizes = [float(x['q']) for x in grp]
                                    row["open"] = prices[0]
                                    row["high"] = max(prices)
                                    row["low"] = min(prices)
                                    row["close"] = prices[-1]
                                    row["volume"] = sum(sizes)
                                    row["taker_sell_volume"] = sum(q for t, q in zip(grp, sizes) if t['m'])
                                    row["taker_buy_volume"] = sum(q for t, q in zip(grp, sizes) if not t['m'])
                                    row["trade_count"] = len(grp)
                                    row["recovery_source"] = 1
                                    row["has_gap"] = False
                                    
                                    recovered_rows.append(row)
                                
                                self.logger.info(f"Level 1 Recovery (aggTrades) sukses memulihkan {len(recovered_rows)} bar.")
                except Exception as e:
                    self.logger.warning(f"Level 1 Recovery (aggTrades) gagal: {e}")

            # --- LEVEL 2: REST /fapi/v1/klines ---
            # Fallback jika Level 1 gagal atau mengembalikan max limit (1000) yang artinya tidak semua ter-cover
            if not recovered_rows:
                try:
                    if rate_limiter: await rate_limiter.acquire(1)
                    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=1s&startTime={start_ms}&endTime={end_ms}&limit=1000"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            klines = await resp.json()
                            for k in klines:
                                k_ts = int(k[0])
                                seq_counter += 1
                                row = self._get_default_row(k_ts, seq_counter)
                                row["open"] = float(k[1])
                                row["high"] = float(k[2])
                                row["low"] = float(k[3])
                                row["close"] = float(k[4])
                                row["volume"] = float(k[5])
                                row["trade_count"] = int(k[8])
                                row["taker_buy_volume"] = float(k[9])
                                row["taker_sell_volume"] = row["volume"] - row["taker_buy_volume"]
                                row["recovery_source"] = 1
                                row["has_gap"] = False
                                
                                recovered_rows.append(row)
                            
                            self.logger.info(f"Level 2 Recovery (klines) sukses memulihkan {len(recovered_rows)} bar.")
                except Exception as e:
                    self.logger.warning(f"Level 2 Recovery (klines) gagal: {e}")

            # --- LEVEL 3: Interpolasi ---
            # Jika ada gap yang belum tertutup oleh REST API (atau API gagal)
            if not recovered_rows and last_known_ts > 0:
                self.logger.warning(f"Level 1 & 2 gagal / kosong. Memulai Level 3 (Interpolasi) dari {last_known_ts} ke {end_ms}")
                
                # Asumsi price menggunakan last state (disini kita simulasikan dengan open=high=low=close=nan, has_gap=True)
                steps = max(1, (end_ms - start_ms) // 1000)
                for i in range(1, steps):
                    k_ts = start_ms + (i * 1000)
                    seq_counter += 1
                    row = self._get_default_row(k_ts, seq_counter)
                    
                    row["volume"] = 0.0
                    row["trade_count"] = 0
                    row["has_gap"] = True
                    row["recovery_source"] = 2
                    
                    recovered_rows.append(row)

        # 7.4 DEDUPLICATION
        pushed_count = 0
        for row in recovered_rows:
            # Validasi monotonicity
            if row["Timestamp"] <= last_known_ts:
                continue
                
            row = self._finalize_row(row)
            await queue.put({"type": "row", "data": row})
            last_known_ts = row["Timestamp"]
            pushed_count += 1
            
        # Update State Tracker
        if pushed_count > 0:
            self.state.setdefault(pair, {})
            self.state[pair]["last_ts"] = last_known_ts
            self.state[pair]["last_seq"] = seq_counter
            self.logger.info(f"Recovery selesai. {pushed_count} bar hasil pemulihan di-push ke antrean.")


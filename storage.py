import os
import time
import math
import asyncio
import logging
import gzip
import shutil
import glob
from datetime import datetime, timezone
from typing import Dict, Any, List

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pcsv

# 6.1 SCHEMA PYARROW (WAJIB EXPLICIT)
OUTPUT_SCHEMA = pa.schema([
    # IDENTIFIERS
    ("Timestamp", pa.int64()), ("local_timestamp_ms", pa.int64()), 
    ("clock_skew_ms", pa.int32()), ("sequence_id", pa.int64()), ("is_warmup", pa.bool_()),
    # OHLCV
    ("open", pa.float32()), ("high", pa.float32()), ("low", pa.float32()), 
    ("close", pa.float32()), ("volume", pa.float32()), ("taker_buy_volume", pa.float32()), 
    ("taker_sell_volume", pa.float32()), ("trade_count", pa.int32()), 
    ("buy_trade_count", pa.int32()), ("sell_trade_count", pa.int32()),
    # ORDER BOOK
    ("best_bid", pa.float32()), ("best_ask", pa.float32()), ("spread", pa.float32()),
    ("mid_price", pa.float32()), ("microprice", pa.float32()), ("weighted_mid_price", pa.float32()),
    ("bid_depth_top5", pa.float32()), ("bid_depth_top10", pa.float32()), ("bid_depth_top20", pa.float32()),
    ("ask_depth_top5", pa.float32()), ("ask_depth_top10", pa.float32()), ("ask_depth_top20", pa.float32()),
    ("cumulative_bid_volume", pa.float32()), ("cumulative_ask_volume", pa.float32()),
    ("order_book_imbalance", pa.float32()), ("volume_imbalance", pa.float32()),
    ("depth_imbalance", pa.float32()), ("slope_bid", pa.float32()), ("slope_ask", pa.float32()),
    ("slope_imbalance", pa.float32()), ("liquidity_pressure", pa.float32()),
    ("queue_imbalance", pa.float32()), ("order_book_pressure_ratio", pa.float32()),
    # TRADE FLOW
    ("aggressive_buy_volume", pa.float32()), ("aggressive_sell_volume", pa.float32()),
    ("delta_volume", pa.float32()), ("cumulative_delta", pa.float32()),
    ("trade_intensity", pa.float32()), ("avg_trade_size", pa.float32()),
    ("max_trade_size", pa.float32()), ("signed_trade_flow", pa.float32()),
    ("trade_burst_metric", pa.float32()), ("trade_arrival_rate", pa.float32()),
    # PRICE ACTION
    ("log_return", pa.float32()), ("realized_volatility", pa.float32()),
    ("parkinson_volatility", pa.float32()), ("bipower_variance", pa.float32()),
    ("momentum_10s", pa.float32()), ("price_acceleration", pa.float32()),
    ("price_impact", pa.float32()), ("micro_trend", pa.int8()), ("directional_pressure", pa.float32()),
    # LIQUIDITY
    ("effective_spread", pa.float32()), ("quoted_spread", pa.float32()),
    ("realized_spread", pa.float32()),  # Nullable by default
    ("kyle_lambda", pa.float32()), ("amihud_illiquidity", pa.float32()),
    ("resiliency_metric", pa.float32()), ("liquidity_vacuum", pa.bool_()),
    # TOXICITY
    ("vpin", pa.float32()), ("toxicity_score", pa.float32()),
    ("adverse_selection_metric", pa.float32()),  # Nullable
    # FUTURES
    ("funding_rate", pa.float32()), ("mark_price", pa.float32()),
    ("index_price", pa.float32()), ("mark_index_spread", pa.float32()),
    ("open_interest", pa.float32()), ("oi_change_rate", pa.float32()),
    ("long_short_ratio", pa.float32()), ("predicted_funding_rate", pa.float32()),
    # REGIME
    ("volatility_regime", pa.int8()), ("trend_regime", pa.int8()),
    ("ranging_regime", pa.bool_()), ("entropy", pa.float32()), ("hurst_exponent", pa.float32()),
    # TIME
    ("second_of_minute", pa.int8()), ("minute_of_hour", pa.int8()),
    ("hour_of_day", pa.int8()), ("day_of_week", pa.int8()), ("session_marker", pa.int8()),
    # QUALITY
    ("book_update_count", pa.int32()), ("trade_msg_count", pa.int32()),
    ("has_gap", pa.bool_()), ("recovery_source", pa.int8()), ("row_checksum", pa.int64())
])


class StorageEngine:
    def __init__(self, shared_state: Dict[str, Any], config: Dict[str, Any]):
        self.shared_state = shared_state
        self.config = config
        self.data_dir = config.get("storage", {}).get("data_dir", "./data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.buffers: Dict[str, List[Dict[str, Any]]] = {pair: [] for pair in shared_state.keys()}
        self.last_flush: Dict[str, float] = {pair: time.time() for pair in shared_state.keys()}
        self.last_ts: Dict[str, int] = {pair: 0 for pair in shared_state.keys()}
        self.current_date = datetime.now(timezone.utc).date()
        self.logger = logging.getLogger("storage")
        self.flush_lock = asyncio.Lock()

    def _validate_row(self, pair: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """Bagian 9: Data Quality Constraints Validator"""
        
        # PERBAIKAN FATAL ERROR: Copy dict agar perubahan nilai menjadi NaN 
        # tidak membocorkan state ke dalam bar_buffer utama di memori features.py
        row = row.copy()
        
        last_ts = self.last_ts[pair]
        ts = row["Timestamp"]
        
        violation = None
        if row["open"] <= 0 or row["high"] <= 0 or row["low"] <= 0 or row["close"] <= 0:
            violation = "Price <= 0"
        elif row["high"] < row["low"]:
            violation = "High < Low"
        elif row["volume"] < 0:
            violation = "Volume < 0"
        elif row["spread"] < 0:
            violation = "Spread < 0"
        elif ts <= last_ts:
            violation = "Timestamp tidak monotonic increasing"

        if violation:
            self.logger.error(f"Kualitas Data Dilanggar pada {pair} ({violation}). Row diconvert ke NaN.")
            # Set price/volume fields ke NaN sesuai spesifikasi
            for f in ["open", "high", "low", "close", "volume", "taker_buy_volume", "taker_sell_volume", "best_bid", "best_ask"]:
                row[f] = float('nan')
            row["has_gap"] = True
            
        self.last_ts[pair] = max(last_ts, ts)
        return row

    def _sync_write_parquet(self, pair: str, date_str: str, table: pa.Table):
        """6.2 WRITE PATTERN - WAL Parquet (Dijalankan di thread terpisah)"""
        fname = os.path.join(self.data_dir, f"{pair}_{date_str}.parquet")
        tmp_fname = os.path.join(self.data_dir, f"{pair}_{date_str}.tmp.parquet")
        
        try:
            if os.path.exists(fname):
                old_table = pq.read_table(fname)
                table = pa.concat_tables([old_table, table])
                
            pq.write_table(table, tmp_fname, compression='zstd', row_group_size=10000)
            os.replace(tmp_fname, fname)
        except Exception as e:
            self.logger.error(f"Gagal menulis parquet WAL untuk {pair}: {e}")

    def _sync_write_csv(self, pair: str, date_str: str, table: pa.Table):
        """Debug CSV, Mode Append"""
        fname = os.path.join(self.data_dir, f"{pair}_{date_str}.csv")
        try:
            write_options = pcsv.WriteOptions(include_header=not os.path.exists(fname))
            with open(fname, "ab") as f:
                pcsv.write_csv(table, f, write_options=write_options)
        except Exception as e:
            self.logger.error(f"Gagal menulis CSV untuk {pair}: {e}")

    async def _flush_buffer(self, pair: str):
        """Flush memory buffer ke disk (Non-blocking lewat to_thread)"""
        buffer_data = self.buffers[pair]
        if not buffer_data:
            return
            
        self.buffers[pair] = []
        self.last_flush[pair] = time.time()
        
        # Ekstrak tanggal dari timestamp record pertama di buffer
        date_str = datetime.fromtimestamp(buffer_data[0]["Timestamp"]/1000, tz=timezone.utc).strftime("%Y%m%d")
        
        try:
            # Transform dictionary keys to match exact schema to avoid pyarrow infer errors
            arrays = {col.name: [row.get(col.name) for row in buffer_data] for col in OUTPUT_SCHEMA}
            table = pa.Table.from_pydict(arrays, schema=OUTPUT_SCHEMA)
            
            await asyncio.to_thread(self._sync_write_parquet, pair, date_str, table)
            await asyncio.to_thread(self._sync_write_csv, pair, date_str, table)
        except Exception as e:
            self.logger.error(f"Gagal konversi ke PyArrow Table untuk {pair}: {e}")

    def _sync_update_disk_row(self, pair: str, date_str: str, seq_id: int, col: str, val: float):
        """Update row yang sudah ter-flush ke Parquet"""
        fname = os.path.join(self.data_dir, f"{pair}_{date_str}.parquet")
        if not os.path.exists(fname):
            return
            
        try:
            table = pq.read_table(fname)
            pydict = table.to_pydict()
            
            if seq_id in pydict["sequence_id"]:
                idx = pydict["sequence_id"].index(seq_id)
                pydict[col][idx] = val
                
                # Update adverse selection metric otomatis jika ada di payload retroactive
                new_table = pa.Table.from_pydict(pydict, schema=OUTPUT_SCHEMA)
                self._sync_write_parquet(pair, date_str, new_table)
        except Exception as e:
            self.logger.error(f"Gagal update_row di disk untuk {pair}, seq {seq_id}: {e}")

    async def _handle_update_row(self, pair: str, update_req: dict):
        """6.3 UPDATE ROW (untuk realized_spread retroactive)"""
        seq_id = update_req["seq_id"]
        rs_val = update_req["realized_spread"]
        adverse_val = update_req["adverse_selection_metric"]
        
        # Cari di buffer memory terlebih dahulu
        for row in reversed(self.buffers[pair]):
            if row["sequence_id"] == seq_id:
                row["realized_spread"] = rs_val
                row["adverse_selection_metric"] = adverse_val
                return
                
        # Jika tidak ada di buffer (sudah di-flush), jalankan update ke disk
        date_str = self.current_date.strftime("%Y%m%d")
        await asyncio.to_thread(self._sync_update_disk_row, pair, date_str, seq_id, "realized_spread", rs_val)
        await asyncio.to_thread(self._sync_update_disk_row, pair, date_str, seq_id, "adverse_selection_metric", adverse_val)

    def _sync_rotate_storage(self):
        """6.5 STORAGE ROTATION"""
        total_size = sum(os.path.getsize(os.path.join(self.data_dir, f)) 
                         for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f)))
        total_gb = total_size / (1024**3)
        
        # Compress CSV kemarin ke .gz jika > 4.5GB
        if total_gb > 4.5:
            csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
            for csvf in csv_files:
                gzf = f"{csvf}.gz"
                if not os.path.exists(gzf):
                    with open(csvf, 'rb') as f_in, gzip.open(gzf, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(csvf)
                    self.logger.info(f"Storage Rotation: Compressed {csvf} to .gz")

        # Hapus Parquet tertua jika > 5.0GB
        if total_gb > 5.0:
            parquet_files = sorted(glob.glob(os.path.join(self.data_dir, "*.parquet")), key=os.path.getctime)
            for pqf in parquet_files:
                # Pastikan ini bukan file hari ini
                if self.current_date.strftime("%Y%m%d") not in pqf:
                    os.remove(pqf)
                    self.logger.warning(f"Storage Rotation: Deleted oldest parquet {pqf} due to size limit.")
                    break

    async def _handle_daily_rollover(self):
        """6.4 DAILY ROLLOVER (00:00:00 UTC)"""
        new_date = datetime.now(timezone.utc).date()
        if new_date != self.current_date:
            self.logger.info("Mengeksekusi Daily Rollover (00:00:00 UTC)...")
            async with self.flush_lock:
                for pair in self.buffers.keys():
                    await self._flush_buffer(pair)
            
            self.current_date = new_date
            await asyncio.to_thread(self._sync_rotate_storage)
            self.logger.info("Daily Rollover selesai.")

    async def _pair_consumer(self, pair: str, shutdown_event: asyncio.Event):
        """Consumer queue independen untuk setiap pair (Bagian 2.1)"""
        queue: asyncio.Queue = self.shared_state[pair]["queue"]
        
        while not shutdown_event.is_set() or not queue.empty():
            try:
                # Timeout dipakai agar task bisa break jika shutdown_event is_set
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                if msg["type"] == "row":
                    row = self._validate_row(pair, msg["data"])
                    self.buffers[pair].append(row)
                elif msg["type"] == "update":
                    await self._handle_update_row(pair, msg)
                    
                queue.task_done()
                
            except asyncio.TimeoutError:
                pass # Lanjut loop evaluasi
            except Exception as e:
                self.logger.error(f"Error pada queue consumer {pair}: {e}")

            # Flush trigger: setiap 60s ATAU buffer >= 10.000 rows
            time_since_flush = time.time() - self.last_flush[pair]
            if len(self.buffers[pair]) >= 10000 or (time_since_flush >= 60 and len(self.buffers[pair]) > 0):
                async with self.flush_lock:
                    await self._flush_buffer(pair)

    async def flush_all(self):
        """Dipanggil saat Graceful Shutdown (main.py)"""
        async with self.flush_lock:
            for pair in self.buffers.keys():
                await self._flush_buffer(pair)

    async def writer_loop(self, shutdown_event: asyncio.Event):
        """8. Task utama pembaca memori dan pengendali I/O Disk"""
        self.logger.info("Storage Engine siap.")
        
        # Buat task consumer untuk membaca antrean setiap pair
        consumers = [
            asyncio.create_task(self._pair_consumer(pair, shutdown_event), name=f"consumer_{pair}") 
            for pair in self.buffers.keys()
        ]
        
        # Loop utama Storage Engine untuk mengawasi rollover
        while not shutdown_event.is_set():
            await self._handle_daily_rollover()
            await asyncio.sleep(5)
            
        # Tunggu semua consumer selesai membersihkan sisa queue saat shutdown
        await asyncio.gather(*consumers, return_exceptions=True)
        self.logger.info("Storage Engine telah berhenti.")


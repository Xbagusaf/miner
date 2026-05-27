import os
import csv
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

# Daftar kolom output (urutan = urutan kolom di CSV).
# Skema sama persis dengan versi parquet sebelumnya, hanya saja semua
# kolom ditulis sebagai string CSV.
OUTPUT_COLUMNS: List[str] = [
    # IDENTIFIERS
    "Timestamp", "local_timestamp_ms", "clock_skew_ms", "sequence_id", "is_warmup",
    # OHLCV
    "open", "high", "low", "close", "volume", "taker_buy_volume",
    "taker_sell_volume", "trade_count", "buy_trade_count", "sell_trade_count",
    # ORDER BOOK
    "best_bid", "best_ask", "spread", "mid_price", "microprice", "weighted_mid_price",
    "bid_depth_top5", "bid_depth_top10", "bid_depth_top20",
    "ask_depth_top5", "ask_depth_top10", "ask_depth_top20",
    "cumulative_bid_volume", "cumulative_ask_volume",
    "order_book_imbalance", "volume_imbalance",
    "depth_imbalance", "slope_bid", "slope_ask",
    "slope_imbalance", "liquidity_pressure",
    "queue_imbalance", "order_book_pressure_ratio",
    # TRADE FLOW
    "aggressive_buy_volume", "aggressive_sell_volume",
    "delta_volume", "cumulative_delta",
    "trade_intensity", "avg_trade_size",
    "max_trade_size", "signed_trade_flow",
    "trade_burst_metric", "trade_arrival_rate",
    # PRICE ACTION
    "log_return", "realized_volatility",
    "parkinson_volatility", "bipower_variance",
    "momentum_10s", "price_acceleration",
    "price_impact", "micro_trend", "directional_pressure",
    # LIQUIDITY
    "effective_spread", "quoted_spread",
    "realized_spread",
    "kyle_lambda", "amihud_illiquidity",
    "resiliency_metric", "liquidity_vacuum",
    # TOXICITY
    "vpin", "toxicity_score",
    "adverse_selection_metric",
    # FUTURES
    "funding_rate", "mark_price",
    "index_price", "mark_index_spread",
    "open_interest", "oi_change_rate",
    "long_short_ratio", "predicted_funding_rate",
    # REGIME
    "volatility_regime", "trend_regime",
    "ranging_regime", "entropy", "hurst_exponent",
    # TIME
    "second_of_minute", "minute_of_hour",
    "hour_of_day", "day_of_week", "session_marker",
    # QUALITY
    "book_update_count", "trade_msg_count",
    "has_gap", "recovery_source", "row_checksum",
]


class CsvWriter:
    """
    Penulis CSV append-only per pair.
    - Buka satu file handle di awal hari, tulis baris demi baris.
    - Flush ke disk setiap `flush_every` baris agar tidak hilang saat crash.
    - Validasi kualitas data inline (spread<0, harga<=0, dll) → set field ke NaN.
    """

    NAN_FIELDS = (
        "open", "high", "low", "close", "volume",
        "taker_buy_volume", "taker_sell_volume",
        "best_bid", "best_ask", "spread", "mid_price",
        "microprice", "weighted_mid_price",
        "quoted_spread", "effective_spread",
    )

    def __init__(self, pair: str, data_dir: str, flush_every: int = 10):
        self.pair = pair
        self.data_dir = data_dir
        self.flush_every = flush_every
        self.logger = logging.getLogger(f"storage.{pair}")

        self._date_str = self._today_str()
        self._fh = None
        self._writer = None
        self._open_file(self._date_str)

        self.last_ts: int = 0
        self.total_rows: int = 0
        self._unflushed: int = 0

    @staticmethod
    def _today_str() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _filename(self, date_str: str) -> str:
        return os.path.join(self.data_dir, f"{self.pair}_{date_str}.csv")

    def _open_file(self, date_str: str):
        fname = self._filename(date_str)
        is_new = not os.path.exists(fname) or os.path.getsize(fname) == 0
        # line-buffered tidak dipakai — flush manual lebih hemat untuk burst write
        self._fh = open(fname, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore",
        )
        if is_new:
            self._writer.writeheader()
            self._fh.flush()
        self._date_str = date_str

    def _validate(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Validator kualitas data inline. Tidak memodifikasi dict asli."""
        ts = row.get("Timestamp", 0)
        violation = None
        if row.get("open", 0) <= 0 or row.get("high", 0) <= 0 \
                or row.get("low", 0) <= 0 or row.get("close", 0) <= 0:
            violation = "Price <= 0"
        elif row.get("high", 0) < row.get("low", 0):
            violation = "High < Low"
        elif row.get("volume", 0) < 0:
            violation = "Volume < 0"
        elif row.get("spread", 0) < 0:
            violation = "Spread < 0"
        elif ts <= self.last_ts:
            violation = "Timestamp tidak monotonic increasing"

        if violation:
            self.logger.warning(f"Data quality: {violation}. Field harga/volume → NaN.")
            row = dict(row)
            for f in self.NAN_FIELDS:
                row[f] = float("nan")
            row["has_gap"] = True

        self.last_ts = max(self.last_ts, ts)
        return row

    def write_row(self, row: Dict[str, Any]):
        """Tulis satu baris ke CSV. Aman dipanggil dari async context (sync I/O cepat)."""
        # Rotasi harian otomatis: jika tanggal UTC berganti, buka file baru.
        today = self._today_str()
        if today != self._date_str:
            self.logger.info(f"Daily rollover: {self._date_str} → {today}")
            self.close()
            self._open_file(today)

        row = self._validate(row)
        try:
            self._writer.writerow(row)
            self.total_rows += 1
            self._unflushed += 1
            if self._unflushed >= self.flush_every:
                self._fh.flush()
                self._unflushed = 0
        except Exception as e:
            self.logger.error(f"Gagal menulis CSV: {e}")

    def flush(self):
        if self._fh and not self._fh.closed:
            try:
                self._fh.flush()
                self._unflushed = 0
            except Exception as e:
                self.logger.error(f"Gagal flush: {e}")

    def close(self):
        if self._fh and not self._fh.closed:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception as e:
                self.logger.error(f"Gagal close: {e}")


class StorageEngine:
    """
    Container tipis untuk semua CsvWriter. Tugasnya hanya:
    - Inisialisasi writer per pair
    - Menyediakan akses lewat `writers[pair]`
    - Mengeksekusi flush_all() saat shutdown
    - Memantau daily rollover (tapi writer juga melakukannya sendiri)
    """

    def __init__(self, shared_state: Dict[str, Any], config: Dict[str, Any]):
        self.shared_state = shared_state
        self.config = config
        self.data_dir = config.get("storage", {}).get("data_dir", "./data")
        os.makedirs(self.data_dir, exist_ok=True)

        flush_every = int(config.get("storage", {}).get("flush_every", 10))
        self.writers: Dict[str, CsvWriter] = {
            pair: CsvWriter(pair, self.data_dir, flush_every=flush_every)
            for pair in shared_state.keys()
        }
        self.logger = logging.getLogger("storage")

    @property
    def total_rows(self) -> Dict[str, int]:
        return {pair: w.total_rows for pair, w in self.writers.items()}

    @property
    def buffers(self) -> Dict[str, list]:
        """Kompatibilitas dengan monitor lama — buffer effectively 0 karena append langsung."""
        return {pair: [] for pair in self.writers.keys()}

    async def flush_all(self):
        for w in self.writers.values():
            w.flush()

    def close_all(self):
        for w in self.writers.values():
            w.close()

    async def run(self, shutdown_event: asyncio.Event):
        """Loop ringan: cek shutdown setiap 5 detik. Daily rollover ditangani writer sendiri."""
        self.logger.info("Storage Engine (CSV-only) siap.")
        while not shutdown_event.is_set():
            await asyncio.sleep(5)
        await self.flush_all()
        # close_all() dipanggil dari main.py setelah semua IngestionEngine selesai flush_pending()
        self.logger.info("Storage Engine telah berhenti.")

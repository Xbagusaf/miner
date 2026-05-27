import asyncio
import time
import logging
from collections import deque
from typing import Dict, Any, Optional

import aiohttp
import websockets
import orjson

# Asumsi import modul eksternal yang disuplai pada file-file lainnya
from orderbook import OrderBook
from features import FeatureEngine

logger = logging.getLogger("ingestion")

# Konstanta Weight Rate Limit sesuai Bagian 3.4
WEIGHT_DEPTH        = 2
WEIGHT_KLINES       = 1
WEIGHT_AGG_TRADES   = 5
WEIGHT_OPEN_INT     = 1
WEIGHT_PREMIUM_IDX  = 1
WEIGHT_LS_RATIO     = 1

class RateLimiter:
    """
    Implementasi Rate Limiter dengan sliding window 60 detik.
    Hard limit: 1100 weight/menit (safety margin 100 dari batas asli 1200 Binance).
    """
    def __init__(self, limit: int = 1100, window: int = 60):
        self.limit = limit
        self.window = window
        # maxlen sebagai safety net; acquire() sudah membersihkan entry > 60s
        self.requests = deque(maxlen=4096)
        self._lock = asyncio.Lock()

    async def acquire(self, weight: int):
        async with self._lock:
            while True:
                now = time.time()
                # Hapus request yang sudah melewati sliding window 60 detik
                while self.requests and now - self.requests[0][0] > self.window:
                    self.requests.popleft()
                
                current_weight = sum(w for _, w in self.requests)
                if current_weight + weight <= self.limit:
                    self.requests.append((now, weight))
                    break
                else:
                    # Tunggu hingga request terlama expired
                    wait_time = self.window - (now - self.requests[0][0])
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)

class IngestionEngine:
    def __init__(self, pair: str, shared_state: Dict[str, Any], recovery_manager: Any,
                 shutdown_event: asyncio.Event, writer: Any = None):
        self.pair_lower = pair.lower()
        self.pair_upper = pair.upper()
        self.shared_state = shared_state
        self.recovery_manager = recovery_manager
        self.shutdown_event = shutdown_event
        self.writer = writer

        self.rate_limiter = RateLimiter()
        self.shared_state["rate_limiter"] = self.rate_limiter

        self.orderbook = OrderBook()
        self.feature_engine = FeatureEngine(self.pair_upper, self.shared_state, writer=writer)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"ingestion.{self.pair_upper}")
        
        # State Reconnect & Backoff
        self.heartbeat_timeout = 60.0
        self.skew_resync_ms = 2000
        self.downtime_start = 0.0
        self.backoff_sequence = [1, 2, 4, 8, 16, 32, 60]
        self.backoff_idx = 0
        self.last_stable_time = 0.0

        # Antrean di memori selama reconnect (Selama reconnect: JANGAN drop data)
        self.memory_buffer = deque(maxlen=50000) 

    async def _api_get(self, endpoint: str, weight: int, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Wrapper request API REST Binance dengan penanganan 429 dan 418"""
        base_url = "https://fapi.binance.com"
        url = f"{base_url}{endpoint}"
        
        retry_backoff = [1, 2, 4, 8, 16, 32, 60]
        retry_idx = 0

        while not self.shutdown_event.is_set():
            await self.rate_limiter.acquire(weight)
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.read()
                        self.logger.debug(f"API Request {endpoint} consumed weight {weight}")
                        return orjson.loads(data)
                    elif response.status == 429:
                        wait_time = retry_backoff[retry_idx]
                        self.logger.warning(f"HTTP 429 Rate Limit. Backoff: {wait_time}s")
                        await asyncio.sleep(wait_time)
                        retry_idx = min(retry_idx + 1, len(retry_backoff) - 1)
                    elif response.status == 418:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.critical(f"HTTP 418 IP BAN! Stop semua request selama {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
                        self.logger.error(f"HTTP {response.status} pada {endpoint}")
                        await asyncio.sleep(2)
            except Exception as e:
                self.logger.error(f"Error REST API {endpoint}: {e}")
                await asyncio.sleep(2)

    async def fetch_orderbook_snapshot(self):
        """3.3 ORDER BOOK SNAPSHOT INIT"""
        params = {"symbol": self.pair_upper, "limit": 20}
        data = await self._api_get("/fapi/v1/depth", WEIGHT_DEPTH, params)
        if data:
            self.orderbook.init_snapshot(data)
            self.logger.info(f"OrderBook Snapshot Init. lastUpdateId: {self.orderbook.last_update_id}")

    async def _poll_premium_index(self):
        """3.5.a Polling premiumIndex setiap 30 detik"""
        while not self.shutdown_event.is_set():
            data = await self._api_get("/fapi/v1/premiumIndex", WEIGHT_PREMIUM_IDX, {"symbol": self.pair_upper})
            if data:
                self.feature_engine.update_futures("mark_price", float(data.get("markPrice", 0)))
                self.feature_engine.update_futures("index_price", float(data.get("indexPrice", 0)))
                self.feature_engine.update_futures("funding_rate", float(data.get("lastFundingRate", 0)))
                self.feature_engine.update_futures("predicted_funding", float(data.get("lastFundingRate", 0))) 
            await asyncio.sleep(30)

    async def _poll_open_interest(self):
        """3.5.b Polling openInterest setiap 5 detik"""
        while not self.shutdown_event.is_set():
            data = await self._api_get("/fapi/v1/openInterest", WEIGHT_OPEN_INT, {"symbol": self.pair_upper})
            if data:
                self.feature_engine.update_futures("open_interest", float(data.get("openInterest", 0)))
            await asyncio.sleep(5)

    async def _poll_ls_ratio(self):
        """3.5.c Polling longShortRatio setiap 60 detik"""
        while not self.shutdown_event.is_set():
            params = {"symbol": self.pair_upper, "period": "5m", "limit": 1}
            data = await self._api_get("/futures/data/globalLongShortAccountRatio", WEIGHT_LS_RATIO, params)
            if data and isinstance(data, list) and len(data) > 0:
                self.feature_engine.update_futures("long_short_ratio", float(data[0].get("longShortRatio", 0)))
            await asyncio.sleep(60)

    async def _process_message(self, message: str):
        """Proses pesan dari Binance combined stream"""
        local_time_ms = int(time.time() * 1000)
        try:
            payload = orjson.loads(message)
            stream = payload.get("stream", "")
            data = payload.get("data", {})
            
            event_time_ms = data.get("E", local_time_ms)
            clock_skew_ms = self.shared_state.get("clock_skew_ms", 0)
            
            if stream.endswith("@aggTrade"):
                self.feature_engine.add_trade(data)
                await self.feature_engine.tick(event_time_ms, local_time_ms, clock_skew_ms)
                
            elif stream.endswith("@depth20@100ms"):
                self.orderbook.update(data)
                self.feature_engine.update_ob(self.orderbook)
                await self.feature_engine.tick(event_time_ms, local_time_ms, clock_skew_ms)

        except Exception as e:
            self.logger.error(f"Gagal memproses pesan WS: {e}", exc_info=True)

    async def _ws_loop(self):
        """Main WebSocket loop dengan pemisahan rute /public dan /market (Sesuai Binance Update)"""
        
        url_public = f"wss://fstream.binance.com/public/stream?streams={self.pair_lower}@depth20@100ms"
        url_market = f"wss://fstream.binance.com/market/stream?streams={self.pair_lower}@aggTrade"
        
        while not self.shutdown_event.is_set():
            try:
                # Connection Stabil: Reset backoff
                self.last_stable_time = time.time()
                
                if self.downtime_start > 0:
                    downtime_end = time.time()
                    downtime_duration = downtime_end - self.downtime_start
                    
                    # Re-init OrderBook via REST snapshot
                    await self.fetch_orderbook_snapshot()
                    
                    # Panggil RecoveryManager
                    await self.recovery_manager.handle_gap(self.pair_upper, self.downtime_start, downtime_end)
                    
                    self.logger.info(f"Reconnected. Downtime: {downtime_duration:.2f}s")
                    self.downtime_start = 0.0

                self.logger.info(f"Mulai mendengarkan stream [Public & Market] terpisah untuk {self.pair_upper}...")

                # Proses isi memory_buffer selama disconnect (jika ada internal event)
                while self.memory_buffer:
                    msg = self.memory_buffer.popleft()
                    await self._process_message(msg)
                
                # Fungsi internal listener untuk di-spawn menjadi concurrent task
                async def listen_stream(ws_url: str):
                    async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as ws:
                        while not self.shutdown_event.is_set():
                            # Trigger reconnect via exception atau timeout (Heartbeat check)
                            message = await asyncio.wait_for(ws.recv(), timeout=self.heartbeat_timeout)
                            
                            if self.backoff_idx > 0 and (time.time() - self.last_stable_time) > 30:
                                self.backoff_idx = 0

                            if abs(self.shared_state.get("clock_skew_ms", 0)) > self.skew_resync_ms:
                                raise Exception("Clock Skew Exceeded Tolerance")

                            await self._process_message(message)

                # Jalankan stream Public dan Market secara konkuren
                task_public = asyncio.create_task(listen_stream(url_public), name="ws_public")
                task_market = asyncio.create_task(listen_stream(url_market), name="ws_market")

                # Tunggu sampai salah satu dari task gagal/terputus.
                # Jika /public putus, kita tidak bisa memproses /market karena book akan out of sync.
                done, pending = await asyncio.wait(
                    [task_public, task_market],
                    return_when=asyncio.FIRST_EXCEPTION
                )
                
                # Batalkan task yang masih berjalan dan ambil hasil task yang gagal (trigger exceptionnya)
                for task in pending:
                    task.cancel()
                    
                for task in done:
                    task.result() 

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.shutdown_event.is_set():
                    break
                
                if self.downtime_start == 0.0:
                    self.downtime_start = time.time()
                
                wait_time = self.backoff_sequence[self.backoff_idx]
                self.logger.warning(f"WebSocket Terputus ({type(e).__name__}: {e}). Reconnecting Public & Market dalam {wait_time}s...")
                
                await asyncio.sleep(wait_time)
                self.backoff_idx = min(self.backoff_idx + 1, len(self.backoff_sequence) - 1)

    async def run(self):
        """Entry point IngestionEngine yang dijalankan sebagai asyncio.Task di main.py"""
        _timeout = aiohttp.ClientTimeout(total=15)
        self.session = aiohttp.ClientSession(timeout=_timeout)
        
        # Wajib dijalankan saat startup
        await self.fetch_orderbook_snapshot()
        
        # Polling futures dijalankan di background task per pair
        tasks = [
            asyncio.create_task(self._poll_premium_index(), name=f"{self.pair_upper}_poll_premium"),
            asyncio.create_task(self._poll_open_interest(), name=f"{self.pair_upper}_poll_oi"),
            asyncio.create_task(self._poll_ls_ratio(), name=f"{self.pair_upper}_poll_ls"),
            asyncio.create_task(self._ws_loop(), name=f"{self.pair_upper}_ws_loop")
        ]
        
        # Tunggu semua task selesai (saat shutdown_event dipanggil)
        await asyncio.gather(*tasks, return_exceptions=True)

        # Tulis sisa pending bars (5 bar terakhir tanpa realized_spread) ke CSV
        try:
            self.feature_engine.flush_pending()
        except Exception as e:
            self.logger.error(f"Gagal flush pending bars: {e}")

        if self.session and not self.session.closed:
            await self.session.close()

        self.logger.info(f"IngestionEngine untuk {self.pair_upper} telah berhenti.")


import asyncio
import time
import os
import glob
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

class MonitorDashboard:
    """
    Monitor Dashboard berbasis Rich.
    Berjalan sebagai background task dan memperbarui UI setiap 1 detik.
    """
    def __init__(self, shared_state: Dict[str, Any], ingestion_engines: List[Any], storage_engine: Any, recovery_manager: Any):
        self.shared_state = shared_state
        self.ingestion_engines = {eng.pair_upper: eng for eng in ingestion_engines}
        self.storage_engine = storage_engine
        self.recovery_manager = recovery_manager
        self.start_time = time.time()
        
        # Tracking kecepatan bar (Rows/Min) per pair
        self.seq_history: Dict[str, deque] = {
            pair: deque(maxlen=60) for pair in self.shared_state.keys()
        }

    def _calculate_rows_per_min(self, pair: str, current_seq: int) -> int:
        """Menghitung rata-rata baris per menit berdasarkan history sequence"""
        now = time.time()
        history = self.seq_history[pair]
        
        # Hapus data yang lebih tua dari 60 detik
        while history and now - history[0][0] > 60:
            history.popleft()
            
        history.append((now, current_seq))
        
        if len(history) < 2:
            return 0
            
        time_diff = history[-1][0] - history[0][0]
        seq_diff = history[-1][1] - history[0][1]
        
        if time_diff <= 0:
            return 0
            
        return int((seq_diff / time_diff) * 60)

    def _get_disk_usage_gb(self, pair: str) -> float:
        """Menghitung total ukuran file di disk untuk pair tertentu (GB)"""
        data_dir = self.storage_engine.data_dir
        pattern = os.path.join(data_dir, f"{pair}*")
        
        total_bytes = 0
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                total_bytes += os.path.getsize(f)
                
        return total_bytes / (1024 ** 3)

    def _build_layout(self) -> Layout:
        """Membangun tabel UI untuk Rich"""
        layout = Layout()
        
        # 1. HEADER
        current_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        header = Text(f"INSTITUTIONAL-GRADE MARKET MICROSTRUCTURE MINER\n{current_utc}", justify="center", style="bold cyan")
        
        # 2. TABEL PER PAIR
        table = Table(expand=True, show_edge=False, header_style="bold magenta")
        table.add_column("PAIR", justify="left")
        table.add_column("STATUS", justify="center")
        table.add_column("ROWS/MIN", justify="right")
        table.add_column("LAST_TS", justify="right")
        table.add_column("SKEW_MS", justify="right")
        table.add_column("GAPS", justify="right")
        table.add_column("WEIGHT/MIN", justify="right")
        table.add_column("BUFFER_MB", justify="right")
        table.add_column("DISK_GB", justify="right")
        table.add_column("UPTIME", justify="right")

        sum_rows_min = 0
        sum_skew = 0
        sum_gaps = 0
        sum_weight = 0
        sum_buffer_mb = 0.0
        sum_disk_gb = 0.0

        for pair in self.shared_state.keys():
            engine = self.ingestion_engines.get(pair)
            
            # Mendapatkan metrik
            state = self.recovery_manager.state.get(pair, {})
            last_seq = state.get("last_seq", 0)
            last_ts = state.get("last_ts", 0)
            last_ts_str = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).strftime("%H:%M:%S") if last_ts > 0 else "-"
            
            rows_min = self._calculate_rows_per_min(pair, last_seq)
            skew_ms = self.shared_state[pair].get("clock_skew_ms", 0)
            
            # Hitung weight API
            rate_limiter = self.shared_state[pair].get("rate_limiter")
            weight_used = sum(w for _, w in rate_limiter.requests) if rate_limiter else 0
            
            # Hitung ukuran buffer (estimasi 1 row dict ~ 1KB)
            buffer_len = len(self.storage_engine.buffers.get(pair, []))
            buffer_mb = (buffer_len * 1024) / (1024 ** 2)
            
            disk_gb = self._get_disk_usage_gb(pair)
            
            uptime_sec = int(time.time() - self.start_time)
            uptime_str = f"{uptime_sec // 3600:02d}:{(uptime_sec % 3600) // 60:02d}:{uptime_sec % 60:02d}"

            # Menentukan STATUS
            status_text = Text("LIVE", style="bold green")
            if engine:
                if getattr(engine, "downtime_start", 0.0) > 0.0:
                    status_text = Text("RECONNECTING", style="bold yellow")
                elif buffer_len > 2000 or (time.time() - getattr(engine, "last_stable_time", time.time())) < 5:
                    # Jika buffer besar akibat baru terhubung, asumsikan sedang recovering REST api
                    status_text = Text("RECOVERING", style="bold cyan")
                    
            # Jika pair gagal load sama sekali
            if not engine:
                status_text = Text("ERROR", style="bold red")

            # Simulasi kolom gaps (diambil dari logger / fallback default 0 karena kita catch via internal has_gap bool)
            gaps = 0 
            
            table.add_row(
                pair,
                status_text,
                str(rows_min),
                last_ts_str,
                str(skew_ms),
                str(gaps),
                str(weight_used),
                f"{buffer_mb:.2f}",
                f"{disk_gb:.4f}",
                uptime_str
            )
            
            # Akumulasi Summary
            sum_rows_min += rows_min
            sum_skew += abs(skew_ms)
            sum_gaps += gaps
            sum_weight += weight_used
            sum_buffer_mb += buffer_mb
            sum_disk_gb += disk_gb

        # 3. SUMMARY ROW
        total_pairs = len(self.shared_state)
        avg_skew = int(sum_skew / total_pairs) if total_pairs > 0 else 0
        
        table.add_section()
        table.add_row(
            Text("TOTAL", style="bold"),
            "-",
            Text(str(sum_rows_min), style="bold"),
            "-",
            Text(str(avg_skew), style="bold"),
            Text(str(sum_gaps), style="bold"),
            Text(str(sum_weight), style="bold"),
            Text(f"{sum_buffer_mb:.2f}", style="bold"),
            Text(f"{sum_disk_gb:.4f}", style="bold"),
            "-"
        )

        # 4. FOOTER
        footer = Text("Press Ctrl+C to stop gracefully", style="dim italic")

        panel = Panel(
            table,
            title=header,
            subtitle=footer,
            subtitle_align="center",
            border_style="cyan"
        )
        layout.update(panel)
        return layout

    async def run(self, shutdown_event: asyncio.Event):
        """Task loop utama untuk me-render Dashboard"""
        try:
            with Live(self._build_layout(), refresh_per_second=1, screen=True) as live:
                while not shutdown_event.is_set():
                    await asyncio.sleep(1)
                    live.update(self._build_layout())
        except Exception as e:
            # Jika terminal tidak mendukung / Rich gagal load, graceful fail tanpa menjatuhkan worker
            pass


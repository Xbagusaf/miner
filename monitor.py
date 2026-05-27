import asyncio
import time
import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

class MonitorDashboard:
    """
    Monitor Dashboard berbasis Rich.
    Layout vertikal (card per pair) agar nyaman di layar sempit (HP).
    """
    def __init__(self, shared_state: Dict[str, Any], ingestion_engines: List[Any], storage_engine: Any, recovery_manager: Any):
        self.shared_state = shared_state
        self.ingestion_engines = {eng.pair_upper: eng for eng in ingestion_engines}
        self.storage_engine = storage_engine
        self.recovery_manager = recovery_manager
        self.start_time = time.time()
        self.logger = logging.getLogger("monitor")
    def _get_disk_usage_gb(self, pair: str) -> float:
        data_dir = self.storage_engine.data_dir
        total_bytes = 0
        try:
            with os.scandir(data_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.startswith(pair):
                        total_bytes += entry.stat().st_size
        except FileNotFoundError:
            return 0.0
        return total_bytes / (1024 ** 3)

    def _collect_pair_metrics(self, pair: str) -> dict:
        """Kumpulkan semua metrik untuk satu pair (sekali panggil per refresh)."""
        engine = self.ingestion_engines.get(pair)
        state = self.recovery_manager.state.get(pair, {})
        last_ts = state.get("last_ts", 0)
        last_ts_str = (
            datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%H:%M:%S")
            if last_ts > 0 else "-"
        )

        total_rows = self.storage_engine.total_rows.get(pair, 0)
        skew_ms = self.shared_state[pair].get("clock_skew_ms", 0)

        rate_limiter = self.shared_state[pair].get("rate_limiter")
        weight_used = sum(w for _, w in rate_limiter.requests) if rate_limiter else 0

        # CSV append langsung — tidak ada buffer signifikan di memori
        buffer_len = 0
        buffer_mb = 0.0
        disk_gb = self._get_disk_usage_gb(pair)

        uptime_sec = int(time.time() - self.start_time)
        uptime_str = (
            f"{uptime_sec // 3600:02d}:"
            f"{(uptime_sec % 3600) // 60:02d}:"
            f"{uptime_sec % 60:02d}"
        )

        status_str = "LIVE"
        status_style = "bold green"
        border_color = "green"
        if engine:
            if getattr(engine, "downtime_start", 0.0) > 0.0:
                status_str = "RECONNECTING"
                status_style = "bold yellow"
                border_color = "yellow"
            elif buffer_len > 2000 or (
                time.time() - getattr(engine, "last_stable_time", time.time()) < 5
            ):
                status_str = "RECOVERING"
                status_style = "bold cyan"
                border_color = "cyan"
        if not engine:
            status_str = "ERROR"
            status_style = "bold red"
            border_color = "red"

        return {
            "pair": pair,
            "total_rows": total_rows,
            "last_ts_str": last_ts_str,
            "skew_ms": skew_ms,
            "weight_used": weight_used,
            "buffer_mb": buffer_mb,
            "disk_gb": disk_gb,
            "uptime_str": uptime_str,
            "status_str": status_str,
            "status_style": status_style,
            "border_color": border_color,
        }

    def _build_pair_card(self, m: dict) -> Panel:
        """Buat card vertikal (Panel) untuk satu pair."""
        t = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        t.add_column("KEY", style="dim", ratio=1)
        t.add_column("VALUE", ratio=1)

        t.add_row("STATUS",     Text(m["status_str"], style=m["status_style"]))
        t.add_row("TOTAL ROWS", Text(str(m["total_rows"]), style="bold white"))
        t.add_row("LAST_TS",   m["last_ts_str"])
        t.add_row("SKEW_MS",   f'{m["skew_ms"]} ms')
        t.add_row("WEIGHT",    str(m["weight_used"]))
        t.add_row("BUFFER",    f'{m["buffer_mb"]:.2f} MB')
        t.add_row("DISK",      f'{m["disk_gb"]:.4f} GB')
        t.add_row("UPTIME",    m["uptime_str"])

        return Panel(
            t,
            title=f"[bold]{m['pair']}[/bold]",
            border_style=m["border_color"],
            expand=True,
        )

    def _build_summary_card(self, all_metrics: list) -> Panel:
        n = len(all_metrics)
        grand_total_rows = sum(m["total_rows"] for m in all_metrics)
        avg_skew = int(sum(abs(m["skew_ms"]) for m in all_metrics) / n) if n > 0 else 0
        total_buf = sum(m["buffer_mb"] for m in all_metrics)
        total_disk = sum(m["disk_gb"] for m in all_metrics)

        t = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        t.add_column("KEY", style="dim", ratio=1)
        t.add_column("VALUE", ratio=1)

        t.add_row("PAIRS",       str(n))
        t.add_row("TOTAL ROWS",  Text(str(grand_total_rows), style="bold white"))
        t.add_row("AVG SKEW",   f"{avg_skew} ms")
        t.add_row("BUFFER",     f"{total_buf:.2f} MB")
        t.add_row("DISK",       f"{total_disk:.4f} GB")

        return Panel(t, title="[bold magenta]SUMMARY[/bold magenta]", border_style="magenta", expand=True)

    def _build_layout(self):
        current_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Outer table sebagai kontainer vertikal (satu kolom)
        outer = Table(show_header=False, box=None, expand=True, padding=0)
        outer.add_column("content", ratio=1)

        # Header
        outer.add_row(Panel(
            Text(current_utc, justify="center", style="bold"),
            title="[bold cyan]MARKET MICROSTRUCTURE MINER[/bold cyan]",
            border_style="cyan",
            expand=True,
        ))

        # Card per pair
        all_metrics = [self._collect_pair_metrics(pair) for pair in self.shared_state.keys()]
        for m in all_metrics:
            outer.add_row(self._build_pair_card(m))

        # Summary
        outer.add_row(self._build_summary_card(all_metrics))

        # Footer
        outer.add_row(Text("Press Ctrl+C to stop gracefully", justify="center", style="dim italic"))

        return outer

    async def run(self, shutdown_event: asyncio.Event):
        """Task loop utama untuk me-render Dashboard."""
        try:
            with Live(self._build_layout(), refresh_per_second=1, screen=False) as live:
                while not shutdown_event.is_set():
                    await asyncio.sleep(1)
                    live.update(self._build_layout())
        except Exception as e:
            self.logger.error(f"Monitor Dashboard error: {e}", exc_info=True)

import asyncio
import argparse
import yaml
import ntplib
import logging
import signal
import sys
from typing import Dict, Any

# 2.1 EVENT LOOP: Menggunakan uvloop sebagai event loop policy
import uvloop

# Import modul yang akan diimplementasikan pada tahap selanjutnya
from ingestion import IngestionEngine
from storage import StorageEngine
from recovery import RecoveryManager
from monitor import MonitorDashboard

# Konfigurasi logging standar
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("miner.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("main")

def check_ntp_sync() -> int:
    """
    Melakukan sinkronisasi NTP (Network Time Protocol) dan mengembalikan 
    clock skew dalam satuan milidetik sesuai spesifikasi Bagian 2.2.
    """
    client = ntplib.NTPClient()
    try:
        response = client.request('pool.ntp.org', version=3, timeout=5)
        skew_ms = int(response.offset * 1000)
        
        if abs(skew_ms) > 2000:
            logger.critical(f"NTP Sync: Clock skew sangat tinggi! Skew: {skew_ms}ms")
        elif abs(skew_ms) > 500:
            logger.warning(f"NTP Sync: Clock skew terdeteksi. Skew: {skew_ms}ms")
        else:
            logger.info(f"NTP Sync: Clock sinkron. Skew: {skew_ms}ms")
            
        return skew_ms
    except Exception as e:
        logger.error(f"Gagal melakukan NTP sync: {e}. Menggunakan asumsi skew = 0ms.")
        return 0

async def shutdown_handler(sig, shutdown_event: asyncio.Event, storage_engine: StorageEngine, recovery_manager: RecoveryManager):
    """
    11. Graceful shutdown: tangkap SIGINT/SIGTERM, flush buffer, tutup koneksi, simpan recovery state.
    """
    signame = sig.name if hasattr(sig, 'name') else sig
    logger.info(f"Menerima sinyal {signame}. Memulai graceful shutdown...")
    
    # Trigger event agar semua task bisa melakukan cleanup masing-masing
    shutdown_event.set()
    
    # Beri toleransi waktu singkat untuk task lain bereaksi
    await asyncio.sleep(2)
    
    logger.info("Menyimpan state recovery ke disk...")
    await recovery_manager.save_state()
    
    logger.info("Flush buffer storage tersisa...")
    await storage_engine.flush_all()
    
    logger.info("Sistem berhasil dihentikan secara aman.")

async def main():
    # 1. Parse CLI argument: --config (default: config.yaml)
    parser = argparse.ArgumentParser(description="Institutional-Grade Market Microstructure Data Miner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # 2. NTP sync check via ntplib
    clock_skew_ms = check_ntp_sync()

    # 3. Load config.yaml
    try:
        with open(args.config, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"Gagal membaca konfigurasi {args.config}: {e}")
        sys.exit(1)

    pairs = config.get("pairs", [])
    if not pairs:
        logger.critical("Tidak ada 'pairs' yang didefinisikan dalam config.yaml.")
        sys.exit(1)

    # Event untuk notifikasi graceful shutdown ke semua child-tasks
    shutdown_event = asyncio.Event()

    # 4. Inisialisasi shared_state dict per pair (tanpa queue — CSV append langsung)
    shared_state = {}
    for pair in pairs:
        shared_state[pair] = {
            "clock_skew_ms": clock_skew_ms,
            "config": config,
            "rate_limiter": None # Akan diinisialisasi oleh IngestionEngine terkait
        }

    # 5. Inisialisasi StorageEngine (CSV-only, satu writer per pair)
    storage_engine = StorageEngine(shared_state, config)

    # 6. Inisialisasi RecoveryManager (terima shutdown_event & writers untuk CSV langsung)
    recovery_manager = RecoveryManager(
        shared_state, config,
        shutdown_event=shutdown_event,
        writers=storage_engine.writers,
    )

    # Memasang Signal Handler untuk Graceful Shutdown (SIGINT, SIGTERM)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(
                    shutdown_handler(s, shutdown_event, storage_engine, recovery_manager)
                )
            )
        except NotImplementedError:
            # Fallback untuk sistem operasi yang tidak mendukung add_signal_handler secara native (misal Windows)
            signal.signal(sig, lambda s, f: loop.create_task(
                shutdown_handler(s, shutdown_event, storage_engine, recovery_manager)
            ))

    tasks = []
    ingestion_engines = []

    # 7. Buat asyncio.Task per pair (IngestionEngine.run) dengan suntikan writer
    for pair in pairs:
        engine = IngestionEngine(
            pair, shared_state[pair], recovery_manager, shutdown_event,
            writer=storage_engine.writers[pair],
        )
        ingestion_engines.append(engine)
        tasks.append(asyncio.create_task(engine.run(), name=f"ingestion_{pair}"))

    # 8. Task tipis StorageEngine — hanya memantau shutdown & flush
    tasks.append(asyncio.create_task(storage_engine.run(shutdown_event), name="storage_runner"))

    # 9. Buat asyncio.Task untuk MonitorDashboard.run (jika enabled)
    if config.get("monitor", {}).get("enabled", True):
        monitor = MonitorDashboard(shared_state, ingestion_engines, storage_engine, recovery_manager)
        tasks.append(asyncio.create_task(monitor.run(shutdown_event), name="monitor_dashboard"))

    logger.info(f"Sistem dimulai dengan {len(pairs)} pairs. Menjalankan engine...")
    
    # 10. asyncio.gather semua tasks
    await asyncio.gather(*tasks, return_exceptions=True)
    
    logger.info("Main event loop selesai.")

if __name__ == "__main__":
    # Set event loop policy ke uvloop tepat sebelum memulai instance asyncio utama
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Ditangkap tanpa error stacktrace panjang karena sudah dihandle graceful shutdown
        pass


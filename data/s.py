import asyncio
import json
import websockets
from datetime import datetime

# Konfigurasi Pair (Ganti sesuai kebutuhan, gunakan huruf KECIL)
PAIR = "hypeusdt"

# Kode ANSI untuk warna terminal
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"

async def monitor_live_spread():
    url = f"wss://fstream.binance.com/ws/{PAIR}@bookTicker"
    
    print(f"{CYAN}================================================={RESET}")
    print(f"{CYAN} Menginisiasi Live Spread Monitor: {PAIR.upper()}{RESET}")
    print(f"{CYAN} Menunggu data dari Binance WebSocket...{RESET}")
    print(f"{CYAN}================================================={RESET}\n")
    
    print(f"| Waktu (Lokal)   | Best Bid  | Best Ask  | Spread    | Status")
    print("-" * 65)

    try:
        async with websockets.connect(url) as ws:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                # bookTicker payload menggunakan 'b' untuk bid dan 'a' untuk ask
                best_bid = float(data.get('b', 0))
                best_ask = float(data.get('a', 0))
                
                spread = best_ask - best_bid
                
                now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Pewarnaan Kondisi Spread
                if spread < 0:
                    status = f"{RED}CROSSED (SPREAD NEGATIF!){RESET}"
                    spread_color = RED
                elif spread == 0:
                    status = f"{YELLOW}LOCKED (SPREAD NOL){RESET}"
                    spread_color = YELLOW
                else:
                    status = f"{GREEN}NORMAL{RESET}"
                    spread_color = GREEN
                    
                print(f"| {now} | {best_bid:<9} | {best_ask:<9} | {spread_color}{spread:<9.4f}{RESET} | {status}")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"\n{RED}Koneksi WebSocket terputus.{RESET}")
    except Exception as e:
        print(f"\n{RED}Terjadi Error: {e}{RESET}")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Monitoring dihentikan oleh user.{RESET}")

if __name__ == "__main__":
    try:
        asyncio.run(monitor_live_spread())
    except KeyboardInterrupt:
        pass


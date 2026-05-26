from typing import Dict, List, Tuple

class OrderBook:
    """
    Manajemen state memori untuk Order Book Futures Binance.
    Menangani inisialisasi dari REST snapshot dan update inkremental dari WebSocket.
    """
    
    def __init__(self):
        # 4.1 STATE
        self.bids: Dict[float, float] = {}  # price → qty
        self.asks: Dict[float, float] = {}  # price → qty
        self.last_update_id: int = 0
        self.update_count: int = 0

    def init_snapshot(self, data: dict):
        """
        4.2 SNAPSHOT INIT
        Menerima data dari REST response (/fapi/v1/depth).
        """
        self.bids.clear()
        self.asks.clear()

        # Parse bids
        for price_str, qty_str in data.get("bids", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty > 0:
                self.bids[price] = qty

        # Parse asks
        for price_str, qty_str in data.get("asks", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty > 0:
                self.asks[price] = qty

        # Set last_update_id dan reset update_count
        self.last_update_id = int(data.get("lastUpdateId", 0))
        self.update_count = 0

    def update(self, data: dict):
        """
        4.3 SNAPSHOT REPLACEMENT untuk @depth20@100ms.
        Stream ini mengirim TOP-20 level terbaru secara penuh setiap 100ms,
        bukan incremental. Level lama HARUS dihapus, bukan diakumulasi.
        Akumulasi level lama menyebabkan crossed book (bid > ask → spread < 0).
        """
        current_u = int(data.get("u", 0))

        # Abaikan jika event update ini lebih tua atau sama dengan update terakhir
        if current_u <= self.last_update_id:
            return

        # Hapus semua level lama — depth20 adalah full snapshot, bukan incremental
        self.bids.clear()
        self.asks.clear()

        # Isi ulang dari snapshot terbaru
        for price_str, qty_str in data.get("b", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty > 0:
                self.bids[price] = qty

        for price_str, qty_str in data.get("a", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty > 0:
                self.asks[price] = qty

        # Update referensi ID dan increment counter
        self.last_update_id = current_u
        self.update_count += 1

    def get_bids(self, n: int = 20) -> List[Tuple[float, float]]:
        """
        4.4 GET SORTED LEVELS (BIDS)
        Return array of tuples (price, qty) sorted descending by price, max n levels.
        """
        # Bids diurutkan dari harga tertinggi ke terendah (descending)
        return sorted(self.bids.items(), key=lambda item: item[0], reverse=True)[:n]

    def get_asks(self, n: int = 20) -> List[Tuple[float, float]]:
        """
        4.4 GET SORTED LEVELS (ASKS)
        Return array of tuples (price, qty) sorted ascending by price, max n levels.
        """
        # Asks diurutkan dari harga terendah ke tertinggi (ascending)
        return sorted(self.asks.items(), key=lambda item: item[0])[:n]

    def reset_update_count(self) -> int:
        """
        Simpan nilai update_count saat ini, reset ke 0, lalu return nilai yang lama.
        Dipanggil setiap kali bar 1 detik selesai dievaluasi.
        """
        old_count = self.update_count
        self.update_count = 0
        return old_count


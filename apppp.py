import cv2
import numpy as np
import time
import queue
import threading
from ultralytics import YOLO
import easyocr

# ===================== KONFIGURASI =====================
YOLO_MODEL_PATH = "data.pt"
KELAS_DETEKSI = [2]   # index kelas plat nomor
OCR_SKIP = 15         # OCR hanya dijalankan setiap 15 frame
TARGET_SIZE = (640, 640)
RTSP_URL = "rtsp://admin:itinl123@192.168.1.18:554/Streaming/Channels/101"
# =======================================================

# ===================== INISIALISASI =====================
print("[INFO] Load model YOLO...")
model = YOLO(YOLO_MODEL_PATH).to("cpu")  # Pakai CPU
pembaca_ocr = easyocr.Reader(['en'], gpu=False)  # OCR di CPU

frame_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

plat_terakhir = ""
jumlah_frame = 0
# =======================================================

def potong_plat(gambar, bbox):
    """Potong area plat dari bounding box YOLO"""
    x1, y1, x2, y2 = map(int, bbox)
    tinggi, lebar = gambar.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, lebar - 1)
    y1, y2 = np.clip([y1, y2], 0, tinggi - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    potongan = gambar[y1:y2, x1:x2]
    if potongan.size == 0:
        return None
    abu = cv2.cvtColor(potongan, cv2.COLOR_BGR2GRAY)
    return abu

def bersihkan_hasil_ocr(hasil_ocr):
    """Bersihkan hasil OCR agar sesuai format plat"""
    import re
    teks_plat = ""
    for _, teks, _ in hasil_ocr:
        teks_bersih = re.sub(r'[^A-Z0-9]', '', teks.upper())
        if len(teks_bersih) > len(teks_plat):
            teks_plat = teks_bersih
    return teks_plat

# ===================== THREAD 1: Ambil Video =====================
def ambil_video():
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Batasi FPS jadi 15
    if not cap.isOpened():
        print("[ERROR] Gagal buka kamera")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Resize biar YOLO lebih cepat
        frame_resized = cv2.resize(frame, TARGET_SIZE)

        # Masukkan frame ke antrian, hindari penuh
        if not frame_queue.full():
            frame_queue.put(frame_resized)

    cap.release()

# ===================== THREAD 2: Deteksi YOLO + OCR =====================
def deteksi_yolo():
    global jumlah_frame, plat_terakhir
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        jumlah_frame += 1
        hasil_deteksi = next(model(frame, classes=KELAS_DETEKSI, stream=True), None)

        plat_ditemukan = ""
        if hasil_deteksi is not None and len(hasil_deteksi.boxes) > 0:
            for kotak in hasil_deteksi.boxes:
                cls = int(kotak.cls.cpu().item())
                if cls in KELAS_DETEKSI:
                    koordinat = kotak.xyxy[0].cpu().numpy()
                    crop_plat = potong_plat(frame, koordinat)
                    if crop_plat is not None:
                        if jumlah_frame % OCR_SKIP == 0:
                            hasil = pembaca_ocr.readtext(crop_plat)
                            plat_ditemukan = bersihkan_hasil_ocr(hasil)
                            if plat_ditemukan:
                                plat_terakhir = plat_ditemukan
                        break

        # Kirim hasil YOLO ke antrian untuk ditampilkan
        if not result_queue.full():
            result_queue.put((frame, hasil_deteksi, plat_terakhir))

# ===================== THREAD 3: Tampilkan Video =====================
def tampilkan_video():
    fps_waktu = time.time()
    fps_counter = 0
    fps_text = "FPS: 0"

    while not stop_event.is_set():
        try:
            frame, hasil_deteksi, plat = result_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Gambar hasil deteksi
        if hasil_deteksi is not None:
            frame = hasil_deteksi.plot()

        # Tambahkan teks plat
        cv2.putText(frame, f"Plat: {plat if plat else '-'}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Hitung FPS
        fps_counter += 1
        if time.time() - fps_waktu >= 1:
            fps_text = f"FPS: {fps_counter}"
            fps_counter = 0
            fps_waktu = time.time()

        cv2.putText(frame, fps_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # Tampilkan video
        cv2.imshow("Deteksi Plat Nomor", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
            stop_event.set()
            break

    cv2.destroyAllWindows()

# ===================== MAIN =====================
if __name__ == "__main__":
    t1 = threading.Thread(target=ambil_video, daemon=True)
    t2 = threading.Thread(target=deteksi_yolo, daemon=True)
    t3 = threading.Thread(target=tampilkan_video, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("[INFO] Selesai, semua thread dimatikan.")

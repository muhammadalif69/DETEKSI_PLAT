import os
import cv2
import numpy as np
import time
import collections
import pyautogui
from ultralytics import YOLO
import torch
import easyocr
import re
from threading import Thread, Lock
import queue
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# ===================== KONFIGURASI UTAMA =====================
layar_w, layar_h = pyautogui.size()
panel_w, panel_h = layar_w // 2, layar_h // 2
UKURAN_TARGET = (panel_w, panel_h)

YOLO_MODEL_PATH = "data.pt"
KELAS_DETEKSI = [2]  # kelas plat nomor
WAKTU_TIMEOUT = 10

CROP_W = int(panel_w * 0.7)
CROP_H = int(panel_h * 0.3)
OCR_SKIP = 15  # OCR dijalankan setiap 15 frame

API_HOST = "0.0.0.0"
API_PORT = 8000

FAKTOR_ZOOM = [1.0, 1.0, 1.0, 1.0]  # Zoom per kamera
LANGKAH_ZOOM = 0.1
ZOOM_MIN = 1.0
ZOOM_MAKS = 3.0
kamera_terakhir_direset = None

# Performance tuning for CPU
NUM_TORCH_THREADS = min(20, max(1, (os.cpu_count() or 1) - 2))  # adjust conservatively

# ============================================================

# ===================== KONFIGURASI KAMERA =====================
KAMERA_LIST = [
    {"ip": "192.168.1.18", "username": "admin", "password": "itinl123"},
    {"ip": "192.168.1.61", "username": "admin", "password": "global123"},
    {"ip": "192.168.1.64", "username": "admin", "password": "itinl2025"},
    # Tambahkan kamera lain di sini (maks 4 agar layout 2x2 tetap rapi)
]
# ============================================================

# ===================== VARIABEL GLOBAL =====================
plat_terakhir = ""
waktu_plat_terakhir = 0.0
frame_terbaru = None
frame_per_kamera = [None] * len(KAMERA_LIST)  # frame individual
kunci_status = Lock()
# ============================================================

# ===================== FUNGSI PEMBANTU =====================
def buka_rtsp(ip: str, username: str, password: str, port: int = 554):
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/101"
    print(f"[INFO] Membuka koneksi ke {rtsp_url}")
    kamera = cv2.VideoCapture(rtsp_url)
    if not kamera.isOpened():
        print(f"[ERROR] Gagal koneksi ke kamera {ip} (cek username/password)")
        raise ConnectionError(f"Gagal koneksi RTSP: {rtsp_url}")
    return kamera

def buka_kamera(kap):
    if kap is None or not kap.isOpened():
        print("[WARNING] Kamera gagal dibuka")
        return None
    kap.set(cv2.CAP_PROP_FRAME_WIDTH, UKURAN_TARGET[0])
    kap.set(cv2.CAP_PROP_FRAME_HEIGHT, UKURAN_TARGET[1])
    try:
        kap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    time.sleep(1)
    return kap

def buat_frame_kosong(teks="TIDAK ADA KAMERA"):
    kosong = np.zeros((UKURAN_TARGET[1], UKURAN_TARGET[0], 3), dtype=np.uint8)
    tulis_teks_tengah(kosong, teks, UKURAN_TARGET[0], UKURAN_TARGET[1],
                      skala=2, warna=(0, 0, 255), tebal=4)
    return kosong

def tulis_teks_tengah(img, teks, box_w, box_h, font=cv2.FONT_HERSHEY_SIMPLEX,
                      skala=2, warna=(0, 0, 255), tebal=4):
    ukuran_teks, _ = cv2.getTextSize(teks, font, skala, tebal)
    teks_w, teks_h = ukuran_teks
    x = (box_w - teks_w) // 2
    y = (box_h + teks_h) // 2
    cv2.putText(img, teks, (x, y), font, skala, warna, tebal, cv2.LINE_AA)
    return img

def potong_plat(gambar, bbox):
    # bbox: [x1, y1, x2, y2]
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

KODE_WILAYAH_VALID = {
    "A","B","BA","BB","BD","BE","BG","BH","BK","BL","BM","BN","BP","D","DA",
    "DB","DC","DD","DE","DG","DH","DK","DL","DM","DN","DR","DS","DT","DU",
    "E","EB","ED","F","G","H","K","KB","KH","KT","KU","L","M","N","P","R",
    "S","T","V","W","Z"
}

def bersihkan_hasil_ocr(hasil_ocr):
    teks_plat = ""
    for _, teks, _ in hasil_ocr:
        teks_bersih = re.sub(r'[^A-Z0-9]', '', teks.upper())
        if len(teks_bersih) > len(teks_plat):
            teks_plat = teks_bersih
    if not teks_plat:
        return ""
    cocok = re.match(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$", teks_plat)
    if cocok:
        kode, nomor, akhiran = cocok.groups()
        if kode not in KODE_WILAYAH_VALID:
            return teks_plat
        format_plat = f"{kode} {nomor}"
        if akhiran:
            format_plat += f" {akhiran}"
        return format_plat
    return teks_plat
# ============================================================

# ===================== INISIALISASI (CPU TUNING) =====================
# Atur environment untuk paralelisasi CPU
os.environ["OMP_NUM_THREADS"] = str(NUM_TORCH_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_TORCH_THREADS)
torch.set_num_threads(NUM_TORCH_THREADS)

device = "cpu"
print(f"[INFO] Perangkat yang digunakan: {device}, torch threads: {torch.get_num_threads()}")

# Muat model YOLO - pastikan model yang dipakai cocok untuk CPU (mis. yolov8n custom)
model = YOLO(YOLO_MODEL_PATH).to(device)

# EasyOCR inisialisasi (hanya english untuk plate format)
pembaca_ocr = easyocr.Reader(['en'], gpu=False)

sumber_kamera = []
kamera_caps = []
kamera_aktif = []
waktu_frame_terakhir = []

for kam in KAMERA_LIST:
    try:
        cap = buka_rtsp(kam["ip"], kam["username"], kam["password"])
        cap = buka_kamera(cap)
        sumber_kamera.append(kam)
        kamera_caps.append(cap)
        kamera_aktif.append(cap is not None)
        waktu_frame_terakhir.append(time.time())
    except Exception as e:
        print("[WARN] Tidak bisa buka kamera", kam.get("ip"), e)
        sumber_kamera.append(kam)
        kamera_caps.append(None)
        kamera_aktif.append(False)
        waktu_frame_terakhir.append(0.0)

riwayat_fps = collections.deque(maxlen=30)
waktu_fps_sebelumnya = time.time()

antrian_ocr = queue.Queue(maxsize=8)
cache_hasil_ocr = ""
jumlah_frame = 0
# ============================================================

# ===================== WORKER OCR (optimized) =====================
def pekerja_ocr():
    global cache_hasil_ocr, plat_terakhir, waktu_plat_terakhir
    while True:
        gambar_crop = antrian_ocr.get()
        if gambar_crop is None:
            break
        try:
            # Resize ke ukuran kecil tapi mempertahankan rasio agar OCR lebih cepat
            h, w = gambar_crop.shape[:2]
            if w > 400:
                scale = 400.0 / w
                nw = int(w * scale)
                nh = int(h * scale)
                gambar_crop = cv2.resize(gambar_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

            # Praproses sederhana, blur ringan + threshold adaptif jika perlu
            blur = cv2.GaussianBlur(gambar_crop, (3, 3), 0)
            # Opsi thresholding jika kontras rendah (non-destruktif)
            mean = np.mean(blur)
            if mean < 120:
                _, diproses = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                diproses = blur

            hasil = pembaca_ocr.readtext(diproses)
            if hasil:
                bersih = bersihkan_hasil_ocr(hasil)
                if bersih:
                    with kunci_status:
                        cache_hasil_ocr = bersih
                        plat_terakhir = bersih
                        waktu_plat_terakhir = time.time()
        except Exception as e:
            print("[WARN] OCR worker error:", e)
        finally:
            antrian_ocr.task_done()

Thread(target=pekerja_ocr, daemon=True).start()
# ============================================================

# ===================== API FASTAPI =====================
app = FastAPI(title="API DETEKSI PLAT")

@app.get("/")
def root():
    return {"pesan": "API DETEKSI PLAT AKTIF", "device": device}

@app.get("/kesehatan")
def kesehatan():
    status_kamera = all(kamera_aktif)
    return {"ok": True, "model_terload": model is not None, "kamera_terhubung": status_kamera}

@app.get("/plat")
def ambil_plat():
    with kunci_status:
        if not plat_terakhir:
            return JSONResponse(status_code=404, content={"error": "Belum ada plat terbaca"})
        return {"plat": plat_terakhir, "timestamp": waktu_plat_terakhir}

# Endpoint gabungan 4 kamera
@app.get("/video")
def stream_video():
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    def hasil():
        while True:
            with kunci_status:
                frame = frame_terbaru.copy() if frame_terbaru is not None else None
            if frame is None:
                time.sleep(0.02)
                continue
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   bytearray(buffer) +
                   b"\r\n")
    return StreamingResponse(hasil(), media_type="multipart/x-mixed-replace; boundary=frame")

# Endpoint per kamera
@app.get("/video/{kamera_id}")
def stream_video_per_kamera(kamera_id: int):
    if kamera_id < 1 or kamera_id > len(KAMERA_LIST):
        return JSONResponse(status_code=404, content={"error": "Kamera tidak ditemukan"})
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    def hasil():
        while True:
            with kunci_status:
                frame = frame_per_kamera[kamera_id - 1]
            if frame is None:
                time.sleep(0.02)
                continue
            ret, buf = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   bytearray(buf) +
                   b"\r\n")
    return StreamingResponse(hasil(), media_type="multipart/x-mixed-replace; boundary=frame")

def mulai_api_dalam_thread():
    def _run():
        uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
    t = Thread(target=_run, daemon=True)
    t.start()
    print(f"[INFO] API berjalan di http://{API_HOST}:{API_PORT}/")
# ============================================================

# ===================== FUNGSI PROSES KAMERA (per-thread) =====================
def proses_kamera(idx):
    global frame_per_kamera, cache_hasil_ocr, jumlah_frame
    kap = kamera_caps[idx]
    last_read = time.time()
    while True:
        if kap is None or not kap.isOpened():
            # coba reconnect
            try:
                print(f"[INFO] mencoba reconnect kamera idx {idx}")
                kam = sumber_kamera[idx]
                kap = buka_rtsp(kam["ip"], kam["username"], kam["password"])
                kap = buka_kamera(kap)
                kamera_caps[idx] = kap
                kamera_aktif[idx] = True
                kamera_caps[idx] = kap
            except Exception:
                kamera_aktif[idx] = False
                frame_per_kamera[idx] = cv2.resize(buat_frame_kosong(f"Kamera {idx+1}: Tidak Aktif"), UKURAN_TARGET)
                time.sleep(1.0)
                continue

        ret, frame = kap.read()
        if not ret or frame is None:
            print(f"[WARN] Kamera {idx+1} tidak mengirim frame")
            kamera_aktif[idx] = False
            try:
                kap.release()
            except Exception:
                pass
            kamera_caps[idx] = None
            time.sleep(0.5)
            continue

        last_read = time.time()
        jumlah_frame += 1

        # Apply zoom (crop + resize) jika diperlukan
        if idx < len(FAKTOR_ZOOM) and FAKTOR_ZOOM[idx] > 1.0:
            try:
                tinggi, lebar = frame.shape[:2]
                baru_w = int(lebar / FAKTOR_ZOOM[idx])
                baru_h = int(tinggi / FAKTOR_ZOOM[idx])
                x1 = max(0, lebar // 2 - baru_w // 2)
                y1 = max(0, tinggi // 2 - baru_h // 2)
                x2 = min(lebar, x1 + baru_w)
                y2 = min(tinggi, y1 + baru_h)
                terpotong = frame[y1:y2, x1:x2]
                if terpotong.size != 0:
                    frame = cv2.resize(terpotong, (lebar, tinggi))
            except Exception as e:
                print("[WARNING] Error zoom:", e)

        # Run YOLO detect (CPU friendly, smaller imgsz)
        try:
            # imgsz diturunkan agar inference lebih cepat di CPU (480 atau 640)
            hasil_list = model.predict(frame, imgsz=640, conf=0.45, device='cpu', classes=KELAS_DETEKSI, verbose=False)
            hasil_deteksi = hasil_list[0] if hasil_list else None
        except Exception as e:
            print("[ERROR] YOLO gagal:", e)
            hasil_deteksi = None

        frame_annotasi = frame.copy()
        crop_slot = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
        tulis_teks_tengah(crop_slot, "Tidak Ada Plat", CROP_W, CROP_H,
                               font=cv2.FONT_HERSHEY_SIMPLEX, skala=1.5,
                               warna=(0, 0, 255), tebal=3)

        plat_ditemukan = False
        if hasil_deteksi is not None and hasattr(hasil_deteksi, "boxes") and len(hasil_deteksi.boxes) > 0:
            # Ambil box pertama (yang paling confidence) untuk kecepatan
            try:
                # 'boxes' tiap item biasanya objek berisi xyxy, cls, conf
                for kotak in hasil_deteksi.boxes:
                    # kotak.xyxy mungkin tensor 1x4
                    koordinat = kotak.xyxy[0].cpu().numpy()
                    cls = int(kotak.cls.cpu().item()) if hasattr(kotak, "cls") else None
                    if cls is not None and cls in KELAS_DETEKSI:
                        crop_plat = potong_plat(frame, koordinat)
                        if crop_plat is not None:
                            # tampilkan crop di pojok frame
                            try:
                                crop_slot = cv2.resize(cv2.cvtColor(crop_plat, cv2.COLOR_GRAY2BGR), (CROP_W, CROP_H))
                            except Exception:
                                crop_slot = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)

                            # hanya enqueue OCR kadang-kadang untuk mengurangi beban
                            if jumlah_frame % OCR_SKIP == 0:
                                # periksa kecerahan sederhana sebelum OCR
                                if np.mean(crop_plat) > 20:
                                    try:
                                        antrian_ocr.put_nowait(crop_plat)
                                    except queue.Full:
                                        # kalau antrian penuh skip
                                        pass
                            plat_ditemukan = True
                            break
            except Exception as e:
                print("[WARN] error saat memproses boxes:", e)

        if not plat_ditemukan:
            with kunci_status:
                cache_hasil_ocr = ""
                # jangan hapus plat_terakhir permanen, biarkan endpoint /plat mengeksekusi timestamp-nya

        # Tempel crop ke frame (kanan bawah)
        tinggi, lebar = frame_annotasi.shape[:2]
        if CROP_H + 20 < tinggi and CROP_W + 20 < lebar:
            y1c, y2c = tinggi - CROP_H - 10, tinggi - 10
            x1c, x2c = lebar - CROP_W - 10, lebar - 10
            frame_annotasi[y1c:y2c, x1c:x2c] = crop_slot

        # tulis teks hasil OCR (snapshot dari cache)
        with kunci_status:
            hasil_cache = cache_hasil_ocr
        teks_tampil = hasil_cache if hasil_cache else "-"
        cv2.putText(frame_annotasi, f"Plat: {teks_tampil}", (10, tinggi - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if hasil_cache else (0, 0, 255), 2)

        cv2.putText(frame_annotasi, f"Kamera {idx+1} (Zoom: {FAKTOR_ZOOM[idx]:.1f}x)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # simpan frame per kamera (resize ke panel)
        try:
            frame_per_kamera[idx] = cv2.resize(frame_annotasi, UKURAN_TARGET)
        except Exception:
            frame_per_kamera[idx] = frame_annotasi

        # timeout handling
        if time.time() - last_read > WAKTU_TIMEOUT:
            print(f"[INFO] Kamera {idx+1} timeout.")
            kamera_aktif[idx] = False
            try:
                kap.release()
            except Exception:
                pass
            kamera_caps[idx] = None
            frame_per_kamera[idx] = cv2.resize(buat_frame_kosong(f"Kamera {idx+1}: Timeout"), UKURAN_TARGET)
            time.sleep(0.5)

# ============================================================

# ===================== LOOP UTAMA (kombinasi dan UI) =====================
def loop_utama():
    global frame_terbaru, waktu_fps_sebelumnya, riwayat_fps, jumlah_frame, kamera_terakhir_direset

    # start threads per kamera
    for i in range(len(KAMERA_LIST)):
        t = Thread(target=proses_kamera, args=(i,), daemon=True)
        t.start()

    waktu_fps_sebelumnya = time.time()
    while True:
        # gabungkan frame per kamera
        frames = []
        with kunci_status:
            for i in range(4):
                if i < len(frame_per_kamera) and frame_per_kamera[i] is not None:
                    frames.append(frame_per_kamera[i])
                else:
                    frames.append(buat_frame_kosong(f"Kamera {i+1}: Tidak Aktif"))
        # pad jika kurang dari 4
        while len(frames) < 4:
            frames.append(buat_frame_kosong())

        baris_atas = np.hstack(frames[:2])
        baris_bawah = np.hstack(frames[2:4])
        gabungan = np.vstack([baris_atas, baris_bawah])

        # hitung FPS kasar
        sekarang = time.time()
        dt = sekarang - waktu_fps_sebelumnya
        fps = 1.0 / (dt + 1e-6)
        riwayat_fps.append(fps)
        waktu_fps_sebelumnya = sekarang

        with kunci_status:
            frame_terbaru = gabungan.copy()

        # Tampilkan window (opsional)
        try:
            cv2.namedWindow("Deteksi Multi Kamera", cv2.WND_PROP_FULLSCREEN)
            try:
                cv2.setWindowProperty("Deteksi Multi Kamera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass
            cv2.imshow("Deteksi Multi Kamera", gabungan)
        except Exception:
            # Environment tanpa display (misal server headless), skip imshow
            pass

        tombol = cv2.waitKey(1) & 0xFF
        if tombol == 27:  # ESC keluar
            break
        # Zoom in per kamera
        elif tombol == ord('1'):
            if 0 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[0] = min(ZOOM_MAKS, FAKTOR_ZOOM[0] + LANGKAH_ZOOM)
        elif tombol == ord('2'):
            if 1 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[1] = min(ZOOM_MAKS, FAKTOR_ZOOM[1] + LANGKAH_ZOOM)
        elif tombol == ord('3'):
            if 2 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[2] = min(ZOOM_MAKS, FAKTOR_ZOOM[2] + LANGKAH_ZOOM)
        elif tombol == ord('4'):
            if 3 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[3] = min(ZOOM_MAKS, FAKTOR_ZOOM[3] + LANGKAH_ZOOM)
        elif tombol == ord('0'):  # Zoom in semua kamera
            for i in range(len(KAMERA_LIST)):
                if i < len(FAKTOR_ZOOM):
                    FAKTOR_ZOOM[i] = min(ZOOM_MAKS, FAKTOR_ZOOM[i] + LANGKAH_ZOOM)

        # Zoom out per kamera (Shift + angka -> simbol)
        elif tombol == ord('!'):  # Shift+1
            if 0 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[0] = max(ZOOM_MIN, FAKTOR_ZOOM[0] - LANGKAH_ZOOM)
                kamera_terakhir_direset = 0
        elif tombol == ord('@'):  # Shift+2
            if 1 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[1] = max(ZOOM_MIN, FAKTOR_ZOOM[1] - LANGKAH_ZOOM)
                kamera_terakhir_direset = 1
        elif tombol == ord('#'):  # Shift+3
            if 2 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[2] = max(ZOOM_MIN, FAKTOR_ZOOM[2] - LANGKAH_ZOOM)
                kamera_terakhir_direset = 2
        elif tombol == ord('$'):  # Shift+4
            if 3 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[3] = max(ZOOM_MIN, FAKTOR_ZOOM[3] - LANGKAH_ZOOM)
                kamera_terakhir_direset = 3
        elif tombol == ord(')'):  # Shift+0 -> zoom out semua
            for i in range(len(KAMERA_LIST)):
                if i < len(FAKTOR_ZOOM):
                    FAKTOR_ZOOM[i] = max(ZOOM_MIN, FAKTOR_ZOOM[i] - LANGKAH_ZOOM)
            kamera_terakhir_direset = 'semua'

        # Reset zoom (tekan 'r' setelah Shift+angka untuk mereset)
        elif tombol == ord('r') and kamera_terakhir_direset is not None:
            if kamera_terakhir_direset == 'semua':
                for i in range(len(KAMERA_LIST)):
                    if i < len(FAKTOR_ZOOM):
                        FAKTOR_ZOOM[i] = 1.0
            else:
                idx_reset = kamera_terakhir_direset
                if isinstance(idx_reset, int) and idx_reset < len(FAKTOR_ZOOM):
                    FAKTOR_ZOOM[idx_reset] = 1.0
            kamera_terakhir_direset = None

    # selesai loop utama
# ============================================================

# ===================== ENTRYPOINT =====================
if __name__ == "__main__":
    mulai_api_dalam_thread()
    try:
        loop_utama()
    finally:
        # Tutup semua koneksi kamera dan worker OCR dengan rapi
        for kap in kamera_caps:
            if kap:
                try:
                    kap.release()
                except Exception:
                    pass
        cv2.destroyAllWindows()
        # hentikan worker OCR
        try:
            antrian_ocr.put_nowait(None)
        except Exception:
            pass
        print("[INFO] Keluar, semua resource dilepas.")

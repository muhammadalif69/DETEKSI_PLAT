# deteksi_plat_cpu_optimized.py
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
import io
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from concurrent.futures import ThreadPoolExecutor

# ===================== OPTIMASI THREADS (paksa ke CPU, manfaatkan semua core) =====================
# Set environment variables BEFORE libraries that use BLAS are initialized (we already imported numpy/cv2,
# but these env vars still help for new threads/torch calls)
num_cpu = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(num_cpu))
os.environ.setdefault("MKL_NUM_THREADS", str(num_cpu))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_cpu))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(num_cpu))
# Torch thread settings
torch.set_num_threads(num_cpu)
torch.set_num_interop_threads(max(1, min(4, num_cpu//4)))  # avoid oversubscribe
# OpenCV threads
try:
    cv2.setNumThreads(max(1, num_cpu//2))
except Exception:
    pass
# Disable OpenCL in OpenCV (can interfere)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

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
# ============================================================

# ===================== KONFIGURASI KAMERA =====================
KAMERA_LIST = [
    {"ip": "192.168.1.18", "username": "admin", "password": "itinl123"},
    {"ip": "192.168.1.61", "username": "admin", "password": "global123"},
    # Tambahkan kamera lain di sini
]
# ============================================================

# ===================== VARIABEL GLOBAL =====================
plat_terakhir = ""
waktu_plat_terakhir = 0.0
frame_terbaru = None
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
    # Atur target capture ke ukuran layar target utk mengurangi scaling ekstra
    kap.set(cv2.CAP_PROP_FRAME_WIDTH, UKURAN_TARGET[0])
    kap.set(cv2.CAP_PROP_FRAME_HEIGHT, UKURAN_TARGET[1])
    try:
        kap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    time.sleep(0.5)
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

# ===================== INISIALISASI DEVICE & MODEL (PAKSA CPU) =====================
device = "cpu"
print(f"[INFO] Perangkat yang digunakan: {device} (dipaksa CPU)")

# Muat model YOLO — ultralytics biasanya menangani device per panggilan infer,
# tetapi .to('cpu') akan memastikan model berada di CPU jika dukungan ada.
model = YOLO(YOLO_MODEL_PATH).to("cpu")

# EasyOCR pakai GPU = False agar CPU-only
pembaca_ocr = easyocr.Reader(['en'], gpu=False)

# Buka kamera
sumber_kamera = []
for kam in KAMERA_LIST:
    try:
        sumber_kamera.append(buka_rtsp(kam["ip"], kam["username"], kam["password"]))
    except Exception:
        sumber_kamera.append(None)

kamera_caps = [buka_kamera(kap) for kap in sumber_kamera]
kamera_aktif = [kap is not None and kap.isOpened() for kap in kamera_caps]
waktu_frame_terakhir = [time.time()] * len(sumber_kamera)

riwayat_fps = collections.deque(maxlen=30)
waktu_fps_sebelumnya = time.time()

antrian_ocr = queue.Queue(maxsize=64)
cache_hasil_ocr = ""
jumlah_frame = 0
# ============================================================

# ===================== WORKER OCR MULTI (mengurangi bottleneck) =====================
ocr_workers = max(1, min(4, max(1, num_cpu // 8)))  # jangan spawn terlalu banyak; EPYC banyak core
ocr_executor = ThreadPoolExecutor(max_workers=ocr_workers)
stop_ocr = False

def ocr_worker_loop(q: queue.Queue):
    global cache_hasil_ocr, plat_terakhir, waktu_plat_terakhir
    while True:
        item = q.get()
        if item is None:
            break
        try:
            gambar_crop = item
            # Preprocess ringan: resize sedikit & threshold
            diproses = cv2.resize(gambar_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            diproses = cv2.GaussianBlur(diproses, (3, 3), 0)
            _, diproses = cv2.threshold(diproses, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            hasil = pembaca_ocr.readtext(diproses)
            if hasil:
                bersih = bersihkan_hasil_ocr(hasil)
                if bersih:
                    with kunci_status:
                        cache_hasil_ocr = bersih
                        plat_terakhir = bersih
                        waktu_plat_terakhir = time.time()
        except Exception as e:
            print("[WARNING] OCR worker error:", e)
        finally:
            q.task_done()

# Start OCR worker threads
for _ in range(ocr_workers):
    ocr_executor.submit(ocr_worker_loop, antrian_ocr)
# ============================================================

# ===================== API FASTAPI =====================
app = FastAPI(title="API DETEKSI PLAT CPU")

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

@app.get("/frame")
def ambil_frame():
    with kunci_status:
        frame = frame_terbaru.copy() if frame_terbaru is not None else None
    if frame is None:
        return JSONResponse(status_code=404, content={"error": "Belum ada frame"})
    ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        raise HTTPException(status_code=500, detail="Gagal encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.get("/video")
def stream_video():
    def hasil():
        while True:
            with kunci_status:
                frame = frame_terbaru.copy() if frame_terbaru is not None else None
            if frame is None:
                time.sleep(0.01)
                continue
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   bytearray(buffer) +
                   b"\r\n")
    return StreamingResponse(hasil(), media_type="multipart/x-mixed-replace; boundary=frame")

def mulai_api_dalam_thread():
    def _run():
        uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
    t = Thread(target=_run, daemon=True)
    t.start()
    print(f"[INFO] API berjalan di http://{API_HOST}:{API_PORT}/")
# ============================================================

# ===================== LOOP UTAMA (dengan annotasi manual ringan) =====================
def loop_utama():
    global jumlah_frame, cache_hasil_ocr, frame_terbaru, waktu_fps_sebelumnya, FAKTOR_ZOOM, kamera_terakhir_direset

    while True:
        frame_semua = []
        waktu_saat_ini = time.time()
        jumlah_frame += 1

        for idx, kap in enumerate(kamera_caps):
            # reconnect jika perlu
            if kap is None or not kap.isOpened():
                try:
                    kap = buka_rtsp(KAMERA_LIST[idx]["ip"], KAMERA_LIST[idx]["username"], KAMERA_LIST[idx]["password"])
                    kap = buka_kamera(kap)
                    kamera_caps[idx] = kap
                    kamera_aktif[idx] = True
                    waktu_frame_terakhir[idx] = waktu_saat_ini
                except Exception:
                    frame_semua.append(buat_frame_kosong(f"Kamera {idx+1}: Tidak Aktif"))
                    continue

            ret, frame = kap.read()
            if not ret or frame is None:
                print(f"[WARNING] Kamera {idx+1} tidak mengirim frame.")
                kamera_aktif[idx] = False
                try:
                    kap.release()
                except Exception:
                    pass
                kamera_caps[idx] = None
                frame_semua.append(buat_frame_kosong(f"Kamera {idx+1}: Tidak Aktif"))
                continue

            waktu_frame_terakhir[idx] = waktu_saat_ini

            # Zoom crop ringan
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
                        frame = cv2.resize(terpotong, (lebar, tinggi), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    print("[WARNING] Error zoom:", e)

            # Jalankan YOLO (CPU) -- stream=True untuk generator lebih efisien memory
            hasil_deteksi = None
            try:
                gen = model(frame, classes=KELAS_DETEKSI, device="cpu", stream=True, verbose=False)
                hasil_deteksi = next(gen, None)
            except Exception as e:
                print("[ERROR] YOLO gagal:", e)
                hasil_deteksi = None

            frame_annotasi = frame.copy()

            # Annotasi manual (lebih cepat daripada .plot())
            if hasil_deteksi is not None and hasattr(hasil_deteksi, "boxes"):
                try:
                    for kotak in hasil_deteksi.boxes:
                        x1, y1, x2, y2 = map(int, kotak.xyxy[0].cpu().numpy())
                        cls = int(kotak.cls.cpu().item())
                        conf = float(kotak.conf.cpu().item()) if hasattr(kotak, "conf") else 0.0
                        if cls in KELAS_DETEKSI:
                            cv2.rectangle(frame_annotasi, (x1, y1), (x2, y2), (0,255,0), 2)
                            cv2.putText(frame_annotasi, f"Plat {conf:.2f}", (x1, max(15, y1-5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                except Exception:
                    pass

            # OCR plat nomor
            crop_slot = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
            tulis_teks_tengah(crop_slot, "Tidak Ada Plat", CROP_W, CROP_H,
                               font=cv2.FONT_HERSHEY_SIMPLEX, skala=1.5,
                               warna=(0, 0, 255), tebal=3)

            plat_ditemukan = False

            if hasil_deteksi is not None and len(hasil_deteksi.boxes) > 0:
                for kotak in hasil_deteksi.boxes:
                    cls = int(kotak.cls.cpu().item())
                    if cls in KELAS_DETEKSI:
                        koordinat = kotak.xyxy[0].cpu().numpy()
                        crop_plat = potong_plat(frame, koordinat)
                        if crop_plat is not None:
                            try:
                                crop_slot = cv2.resize(cv2.cvtColor(crop_plat, cv2.COLOR_GRAY2BGR), (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
                            except Exception:
                                crop_slot = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
                            # enqueue OCR hanya jika ada space (non-blocking put)
                            if jumlah_frame % OCR_SKIP == 0:
                                try:
                                    antrian_ocr.put_nowait(crop_plat)
                                except queue.Full:
                                    # jika queue penuh, skip (OCR worker sibuk) -> menghindari blocking
                                    pass
                            plat_ditemukan = True
                            break

            if not plat_ditemukan:
                with kunci_status:
                    cache_hasil_ocr = ""
                    plat_terakhir = ""

            # Tempel crop di pojok frame
            tinggi, lebar = frame_annotasi.shape[:2]
            if CROP_H + 20 < tinggi and CROP_W + 20 < lebar:
                y1, y2 = tinggi - CROP_H - 10, tinggi - 10
                x1, x2 = lebar - CROP_W - 10, lebar - 10
                frame_annotasi[y1:y2, x1:x2] = crop_slot

            with kunci_status:
                hasil_cache = cache_hasil_ocr

            teks_tampil = hasil_cache if hasil_cache else "-"
            cv2.putText(frame_annotasi, f"Plat: {teks_tampil}", (10, tinggi - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if hasil_cache else (0, 0, 255), 2)

            cv2.putText(frame_annotasi, f"Kamera {idx+1} (Zoom: {FAKTOR_ZOOM[idx]:.1f}x)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            frame_semua.append(cv2.resize(frame_annotasi, UKURAN_TARGET, interpolation=cv2.INTER_AREA))

            # Kamera timeout
            if waktu_saat_ini - waktu_frame_terakhir[idx] > WAKTU_TIMEOUT:
                print(f"[INFO] Kamera {idx+1} timeout.")
                kamera_aktif[idx] = False
                try:
                    kap.release()
                except Exception:
                    pass
                kamera_caps[idx] = None

        # Jika jumlah frame < 4, tambahkan kosong
        while len(frame_semua) < 4:
            frame_semua.append(buat_frame_kosong())

        baris_atas = np.hstack(frame_semua[:2])
        baris_bawah = np.hstack(frame_semua[2:4])
        gabungan = np.vstack([baris_atas, baris_bawah])

        # Hitung FPS
        sekarang = time.time()
        dt = sekarang - waktu_fps_sebelumnya
        fps = 1.0 / (dt + 1e-6)
        riwayat_fps.append(fps)
        waktu_fps_sebelumnya = sekarang

        with kunci_status:
            frame_terbaru = gabungan.copy()

        cv2.namedWindow("Deteksi Multi Kamera", cv2.WND_PROP_FULLSCREEN)
        try:
            cv2.setWindowProperty("Deteksi Multi Kamera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        cv2.imshow("Deteksi Multi Kamera", gabungan)

        tombol = cv2.waitKey(1) & 0xFF
        if tombol == 27:  # ESC keluar
            break
        # Kontrol zoom (sama seperti sebelumnya)
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
        elif tombol == ord('0'):
            for i in range(len(KAMERA_LIST)):
                if i < len(FAKTOR_ZOOM):
                    FAKTOR_ZOOM[i] = min(ZOOM_MAKS, FAKTOR_ZOOM[i] + LANGKAH_ZOOM)

        # Zoom out (Shift + angka)
        elif tombol == ord('!'):
            if 0 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[0] = max(ZOOM_MIN, FAKTOR_ZOOM[0] - LANGKAH_ZOOM)
                kamera_terakhir_direset = 0
        elif tombol == ord('@'):
            if 1 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[1] = max(ZOOM_MIN, FAKTOR_ZOOM[1] - LANGKAH_ZOOM)
                kamera_terakhir_direset = 1
        elif tombol == ord('#'):
            if 2 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[2] = max(ZOOM_MIN, FAKTOR_ZOOM[2] - LANGKAH_ZOOM)
                kamera_terakhir_direset = 2
        elif tombol == ord('$'):
            if 3 < len(FAKTOR_ZOOM):
                FAKTOR_ZOOM[3] = max(ZOOM_MIN, FAKTOR_ZOOM[3] - LANGKAH_ZOOM)
                kamera_terakhir_direset = 3
        elif tombol == ord(')'):
            for i in range(len(KAMERA_LIST)):
                if i < len(FAKTOR_ZOOM):
                    FAKTOR_ZOOM[i] = max(ZOOM_MIN, FAKTOR_ZOOM[i] - LANGKAH_ZOOM)
            kamera_terakhir_direset = 'semua'

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

    # akhir loop
# ============================================================

# ===================== ENTRYPOINT =====================
if __name__ == "__main__":
    mulai_api_dalam_thread()
    try:
        loop_utama()
    finally:
        # Tutup kamera & OCR worker rapi
        for kap in kamera_caps:
            if kap:
                try:
                    kap.release()
                except Exception:
                    pass
        cv2.destroyAllWindows()
        # hentikan worker OCR secara aman
        for _ in range(ocr_workers):
            try:
                antrian_ocr.put_nowait(None)
            except Exception:
                pass
        ocr_executor.shutdown(wait=True)
        print("[INFO] Keluar, semua resource dilepas.")

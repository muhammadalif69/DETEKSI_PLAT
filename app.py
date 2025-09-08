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
    """
    Membuka koneksi RTSP untuk kamera.
    """
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/101"
    print(f"[INFO] Membuka koneksi ke {rtsp_url}")
    kamera = cv2.VideoCapture(rtsp_url)

    if not kamera.isOpened():
        print(f"[ERROR] Gagal koneksi ke kamera {ip} (cek username/password)")
        raise ConnectionError(f"Gagal koneksi RTSP: {rtsp_url}")
    return kamera

def buka_kamera(kap):
    """Mengatur resolusi dan buffer kamera"""
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
    """Membuat frame kosong jika kamera mati"""
    kosong = np.zeros((UKURAN_TARGET[1], UKURAN_TARGET[0], 3), dtype=np.uint8)
    tulis_teks_tengah(kosong, teks, UKURAN_TARGET[0], UKURAN_TARGET[1],
                      skala=2, warna=(0, 0, 255), tebal=4)
    return kosong

def tulis_teks_tengah(img, teks, box_w, box_h, font=cv2.FONT_HERSHEY_SIMPLEX,
                      skala=2, warna=(0, 0, 255), tebal=4):
    """Menulis teks di tengah gambar"""
    ukuran_teks, _ = cv2.getTextSize(teks, font, skala, tebal)
    teks_w, teks_h = ukuran_teks
    x = (box_w - teks_w) // 2
    y = (box_h + teks_h) // 2
    cv2.putText(img, teks, (x, y), font, skala, warna, tebal, cv2.LINE_AA)
    return img

def potong_plat(gambar, bbox):
    """Memotong plat nomor dari bounding box YOLO"""
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

# Kode wilayah plat Indonesia
KODE_WILAYAH_VALID = {
    "A","B","BA","BB","BD","BE","BG","BH","BK","BL","BM","BN","BP","D","DA",
    "DB","DC","DD","DE","DG","DH","DK","DL","DM","DN","DR","DS","DT","DU",
    "E","EB","ED","F","G","H","K","KB","KH","KT","KU","L","M","N","P","R",
    "S","T","V","W","Z"
}

def bersihkan_hasil_ocr(hasil_ocr):
    """Membersihkan hasil OCR agar sesuai format plat nomor Indonesia"""
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

# ===================== INISIALISASI =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Perangkat yang digunakan: {device}")

model = YOLO(YOLO_MODEL_PATH).to(device)
pembaca_ocr = easyocr.Reader(['en'], gpu=(device == "cuda"))

# Buka semua kamera
sumber_kamera = [
    buka_rtsp(kam["ip"], kam["username"], kam["password"])
    for kam in KAMERA_LIST
]
kamera_caps = [buka_kamera(kap) for kap in sumber_kamera]
kamera_aktif = [kap is not None for kap in kamera_caps]
waktu_frame_terakhir = [time.time()] * len(sumber_kamera)

riwayat_fps = collections.deque(maxlen=30)
waktu_fps_sebelumnya = time.time()

antrian_ocr = queue.Queue()
cache_hasil_ocr = ""
jumlah_frame = 0
# ============================================================

# ===================== WORKER OCR =====================
def pekerja_ocr():
    global cache_hasil_ocr, plat_terakhir, waktu_plat_terakhir
    while True:
        gambar_crop = antrian_ocr.get()
        if gambar_crop is None:
            break
        diproses = cv2.resize(gambar_crop, (0, 0), fx=2.0, fy=2.0)
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

@app.get("/frame")
def ambil_frame():
    with kunci_status:
        frame = frame_terbaru.copy() if frame_terbaru is not None else None
    if frame is None:
        return JSONResponse(status_code=404, content={"error": "Belum ada frame"})
    ret, buf = cv2.imencode('.jpg', frame)
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
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
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

# ===================== LOOP UTAMA =====================
def loop_utama():
    global jumlah_frame, cache_hasil_ocr, frame_terbaru, waktu_fps_sebelumnya, FAKTOR_ZOOM, kamera_terakhir_direset

    while True:
        frame_semua = []
        waktu_saat_ini = time.time()
        jumlah_frame += 1

        for idx, kap in enumerate(kamera_caps):
            # Reconnect kamera jika mati
            if not kamera_aktif[idx]:
                try:
                    kap = buka_rtsp(
                        KAMERA_LIST[idx]["ip"],
                        KAMERA_LIST[idx]["username"],
                        KAMERA_LIST[idx]["password"]
                    )
                    kap = buka_kamera(kap)
                    kamera_caps[idx] = kap
                    kamera_aktif[idx] = True
                    waktu_frame_terakhir[idx] = waktu_saat_ini
                except Exception:
                    frame_semua.append(buat_frame_kosong(f"Kamera {idx+1}: Tidak Aktif"))
                    continue

            # Ambil frame kamera
            ret, frame = kap.read()
            if not ret or frame is None:
                print(f"[WARNING] Kamera {idx+1} tidak mengirim frame.")
                kamera_aktif[idx] = False
                if kap:
                    kap.release()
                kamera_caps[idx] = None
                frame_semua.append(buat_frame_kosong(f"Kamera {idx+1}: Tidak Aktif"))
                continue

            waktu_frame_terakhir[idx] = waktu_saat_ini

            # Zoom per kamera
            if FAKTOR_ZOOM[idx] > 1.0:
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

            # Jalankan YOLO
            try:
                hasil_deteksi_gen = model(frame, classes=KELAS_DETEKSI, device=device, stream=True, verbose=False)
                hasil_deteksi = next(hasil_deteksi_gen)
            except StopIteration:
                hasil_deteksi = None
            except Exception as e:
                print("[ERROR] YOLO gagal:", e)
                hasil_deteksi = None

            frame_annotasi = frame.copy()
            if hasil_deteksi is not None:
                try:
                    frame_annotasi = hasil_deteksi.plot()
                except Exception:
                    frame_annotasi = frame.copy()

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
                        crop_plat = potong_plat(frame_annotasi, koordinat)
                        if crop_plat is not None:
                            try:
                                crop_slot = cv2.resize(cv2.cvtColor(crop_plat, cv2.COLOR_GRAY2BGR), (CROP_W, CROP_H))
                            except Exception:
                                crop_slot = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
                            if jumlah_frame % OCR_SKIP == 0:
                                antrian_ocr.put(crop_plat)
                            plat_ditemukan = True
                            break

            if not plat_ditemukan:
                with kunci_status:
                    cache_hasil_ocr = ""
                    plat_terakhir = ""

            # Tempel hasil OCR di frame
            tinggi, lebar = frame_annotasi.shape[:2]
            if CROP_H + 20 < tinggi and CROP_W + 20 < lebar:
                y1, y2 = tinggi - CROP_H - 10, tinggi - 10
                x1, x2 = lebar - CROP_W - 10, lebar - 10
                frame_annotasi[y1:y2, x1:x2] = crop_slot

            with kunci_status:
                hasil_cache = cache_hasil_ocr

            teks_tampil = hasil_cache if hasil_cache else "-"
            cv2.putText(frame_annotasi, f"Plat: {teks_tampil}", (10, tinggi - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if hasil_cache else (0, 0, 255), 3)

            cv2.putText(frame_annotasi, f"Kamera {idx+1} (Zoom: {FAKTOR_ZOOM[idx]:.1f}x)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            frame_semua.append(cv2.resize(frame_annotasi, UKURAN_TARGET))

            # Kamera timeout
            if waktu_saat_ini - waktu_frame_terakhir[idx] > WAKTU_TIMEOUT:
                print(f"[INFO] Kamera {idx+1} timeout.")
                kamera_aktif[idx] = False
                if kap:
                    kap.release()
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

        # Reset zoom (tekan 'r' setelah Shift+angka untuk mereset kamera yang terakhir di-shift)
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

    # Akhir while loop utama


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
        antrian_ocr.put(None)
        print("[INFO] Keluar, semua resource dilepas.")
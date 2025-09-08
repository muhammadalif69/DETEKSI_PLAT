import cv2
import numpy as np
import time
import collections
import pyautogui
from ultralytics import YOLO
import torch
import easyocr
import re
from camera import rtsp_stream
from threading import Thread, Lock
import queue
import io
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
import uvicorn


# ===================== KONFIGURASI =====================
screen_w, screen_h = pyautogui.size()
panel_w, panel_h = screen_w // 2, screen_h // 2
TARGET_SIZE = (panel_w, panel_h)

YOLO_MODEL_PATH = "data.pt"
DETECTION_CLASSES = [2]  # kelas plat nomor
TIMEOUT_SEC = 10

# Ukuran crop untuk OCR (slot preview kanan-bawah)
CROP_W = int(panel_w * 0.7)
CROP_H = int(panel_h * 0.3)
OCR_SKIP = 15  # OCR dijalankan setiap 5 frame

API_HOST = "0.0.0.0"
API_PORT = 8000

# ------------------ DIGITAL ZOOM -----------------------
ZOOM_FACTOR = 1.0   # default zoom (1.0 = no zoom)
ZOOM_STEP = 0.1     # langkah saat menekan hotkey
MIN_ZOOM = 1.0
MAX_ZOOM = 3.0
# =======================================================

# ===================== GLOBAL STATE (shared with API) =====================
latest_plate = ""          # teks plat hasil OCR terakhir (untuk API /plate)
latest_plate_time = 0.0
latest_frame = None        # combined frame (BGR numpy array) untuk API /frame
state_lock = Lock()        # sinkronisasi akses shared state
# =======================================================


# ===================== FUNGSI PEMBANTU =====================
def open_camera(cap):
    if cap is None or not cap.isOpened():
        print("[WARNING] FAILED TO OPEN CAMERA")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_SIZE[1])
    # try to reduce buffer
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    time.sleep(1)
    return cap


def create_blank_frame(text="NO CAMERA"):
    blank = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
    draw_centered_text(blank, text, TARGET_SIZE[0], TARGET_SIZE[1], scale=2, color=(0, 0, 255), thickness=4)
    return blank

def draw_centered_text(img, text, box_w, box_h, font=cv2.FONT_HERSHEY_SIMPLEX,
                       scale=2, color=(0, 0, 255), thickness=4):
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size

    # Hitung posisi agar teks berada di tengah
    x = (box_w - text_w) // 2
    y = (box_h + text_h) // 2  # +text_h agar baseline tepat di tengah

    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return img


def crop(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)

    # Safety bounds
    h, w = image.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    # Crop gambar
    crop_img = image[y1:y2, x1:x2]
    if crop_img.size == 0:
        return None

    # Konversi ke grayscale biasa
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    return gray


VALID_REGION_CODES = {
    "A","B","BA","BB","BD","BE","BG","BH","BK","BL","BM","BN","BP","D","DA",
    "DB","DC","DD","DE","DG","DH","DK","DL","DM","DN","DR","DS","DT","DU",
    "E","EB","ED","F","G","H","K","KB","KH","KT","KU","L","M","N","P","R",
    "S","T","V","W","Z"
}


def filter_plate_text(ocr_result):
    plate_text = ""
    for _, text, _ in ocr_result:
        text_filtered = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(text_filtered) > len(plate_text):
            plate_text = text_filtered

    if not plate_text:
        return ""

    match = re.match(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$", plate_text)
    if match:
        region, number, suffix = match.groups()
        if region not in VALID_REGION_CODES:
            return plate_text

        formatted = f"{region} {number}"
        if suffix:
            formatted += f" {suffix}"
        return formatted

    return plate_text
# =======================================================


# ===================== INISIALISASI MODEL & OCR =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] DEVICE USE: {device}")

# Muat model YOLO (pastikan path benar)
model = YOLO(YOLO_MODEL_PATH).to(device)

# OCR
reader = easyocr.Reader(['en'], gpu=(device == "cuda"))

# Setup kamera (sesuaikan ip/credentials di rtsp_stream)
camera_sources = [rtsp_stream("192.168.1.61", "101")]
caps = [open_camera(cap) for cap in camera_sources]
last_frame_times = [time.time()] * len(camera_sources)
camera_active = [cap is not None for cap in caps]

fps_history = collections.deque(maxlen=30)
prev_fps_time = time.time()

ocr_queue = queue.Queue()
ocr_result_cache = ""  # cache teks untuk overlay
frame_count = 0
# =======================================================


def ocr_worker():
    global ocr_result_cache, latest_plate, latest_plate_time
    while True:
        crop_img = ocr_queue.get()
        if crop_img is None:
            break

        # Perbesar agar OCR lebih akurat
        processed = cv2.resize(crop_img, (0, 0), fx=2.0, fy=2.0)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result = reader.readtext(processed)

        if result:
            filtered = filter_plate_text(result)
            if filtered:
                with state_lock:
                    ocr_result_cache = filtered    # untuk overlay
                    latest_plate = filtered        # untuk API /plate
                    latest_plate_time = time.time()


# Jalankan OCR worker
Thread(target=ocr_worker, daemon=True).start()


# ===================== FASTAPI (dijalankan di thread) =====================
app = FastAPI(title="PLATE API)")

@app.get("/")
def root():
    return {"message": "PLATE API RUNNING", "device": device}

@app.get("/health")
def health():
    # cek model & kamera minimal
    cam_status = all(camera_active)
    return {"ok": True, "model_loaded": model is not None, "camera_connected": cam_status}

@app.get("/plate")
def get_plate():
    with state_lock:
        if not latest_plate:
            return JSONResponse(status_code=404, content={"error": "No plate read yet"})
        return {"plate": latest_plate, "timestamp": latest_plate_time}

@app.get("/frame")
def get_frame():
    with state_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        return JSONResponse(status_code=404, content={"error": "No frame available yet"})

    # encode to jpeg
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


def start_api_in_thread():
    def _run():
        uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
    t = Thread(target=_run, daemon=True)
    t.start()
    print(f"[INFO] API server starting at http://{API_HOST}:{API_PORT}/")


# =======================================================
# ===================== LOOP UTAMA =======================
# =======================================================
def main_loop():
    global frame_count, ocr_result_cache, latest_frame, prev_fps_time, ZOOM_FACTOR

    while True:
        frames = []
        current_time = time.time()
        frame_count += 1

        for idx, cap in enumerate(caps):
            # Reconnect jika kamera mati
            if not camera_active[idx]:
                try:
                    cap = rtsp_stream("192.168.1.61", "101")
                    cap = open_camera(cap)
                    caps[idx] = cap
                    camera_active[idx] = True
                    last_frame_times[idx] = current_time
                except Exception:
                    frames.append(create_blank_frame(f"Camera {idx+1}: No Camera"))
                    continue

            # Baca frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[WARNING] Kamera {idx+1} tidak mengirim frame.")
                camera_active[idx] = False
                if cap:
                    cap.release()
                caps[idx] = None
                frames.append(create_blank_frame(f"Camera {idx+1}: No Camera"))
                continue

            last_frame_times[idx] = current_time

            # === Digital Zoom (crop tengah lalu resize) ===
            if ZOOM_FACTOR > 1.0:
                try:
                    h, w = frame.shape[:2]
                    new_w = int(w / ZOOM_FACTOR)
                    new_h = int(h / ZOOM_FACTOR)
                    # Pastikan minimal 2x2 crop
                    new_w = max(2, new_w)
                    new_h = max(2, new_h)
                    x1 = max(0, w // 2 - new_w // 2)
                    y1 = max(0, h // 2 - new_h // 2)
                    x2 = min(w, x1 + new_w)
                    y2 = min(h, y1 + new_h)
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size != 0:
                        frame = cv2.resize(cropped, (w, h))
                except Exception as e:
                    print("[WARNING] Zoom error:", e)

            # YOLO inference
            try:
                results_gen = model(frame, classes=DETECTION_CLASSES, device=device, stream=True)
                result = next(results_gen)
            except StopIteration:
                result = None   
            except Exception as e:
                print("[ERROR] YOLO inference error:", e)
                result = None

            annotated = frame.copy()
            if result is not None:
                try:
                    annotated = result.plot()
                except Exception:
                    annotated = frame.copy()

            # Crop & OCR plat (default: no plate)
            slot_crop = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
            draw_centered_text(slot_crop, "No Plate", CROP_W, CROP_H,
                   font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.5, color=(0, 0, 255), thickness=3)

            found_plate_in_this_frame = False

            if result is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls = int(box.cls.cpu().item())
                    if cls in DETECTION_CLASSES:
                        coords = box.xyxy[0].cpu().numpy()
                        crop_plate = crop(annotated, coords)

                        if crop_plate is not None:
                            # tampilan untuk slot (convert to BGR)
                            try:
                                slot_crop = cv2.resize(cv2.cvtColor(crop_plate, cv2.COLOR_GRAY2BGR), (CROP_W, CROP_H))
                            except Exception:
                                try:
                                    slot_crop = cv2.resize(np.stack([crop_plate]*3, axis=-1), (CROP_W, CROP_H))
                                except Exception:
                                    slot_crop = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)

                            # Kirim grayscale langsung ke OCR tiap OCR_SKIP frame
                            if frame_count % OCR_SKIP == 0:
                                ocr_queue.put(crop_plate)

                            found_plate_in_this_frame = True
                            break  # ambil satu plat pertama saja

            # === RESET OCR jika tidak ada deteksi pada frame ini ===
            if not found_plate_in_this_frame:
                with state_lock:
                    ocr_result_cache = ""
                    latest_plate = ""
                    # latest_plate_time tidak diubah

            # Tempel hasil crop ke frame annotated (preview pojok kanan bawah)
            h, w = annotated.shape[:2]
            if CROP_H + 20 < h and CROP_W + 20 < w:
                y1, y2 = h - CROP_H - 10, h - 10
                x1, x2 = w - CROP_W - 10, w - 10
                annotated[y1:y2, x1:x2] = slot_crop

            # Tampilkan hasil OCR (shared cache)
            with state_lock:
                cached = ocr_result_cache

            text_to_show = cached if cached else "-"
            cv2.putText(annotated, f"Plate: {text_to_show}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if cached else (0, 0, 255), 3)

            cv2.putText(annotated, f"Camera {idx+1} (Zoom: {ZOOM_FACTOR:.1f}x)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

            frames.append(cv2.resize(annotated, TARGET_SIZE))

            # Timeout kamera
            if current_time - last_frame_times[idx] > TIMEOUT_SEC:
                print(f"[INFO] Kamera {idx+1} timeout.")
                camera_active[idx] = False
                if cap:
                    cap.release()
                caps[idx] = None

        # Lengkapi frame jadi 4 panel
        while len(frames) < 4:
            frames.append(create_blank_frame())

        top_row = np.hstack(frames[:2])
        bottom_row = np.hstack(frames[2:4])
        combined = np.vstack([top_row, bottom_row])

        # Hitung FPS
        now = time.time()
        dt = now - prev_fps_time
        fps = 1.0 / (dt + 1e-6)
        fps_history.append(fps)
        prev_fps_time = now

        # Update shared latest_frame
        with state_lock:
            latest_frame = combined.copy()

        # Tampilkan hasil
        cv2.namedWindow("Multi Camera Detection", cv2.WND_PROP_FULLSCREEN)
        try:
            cv2.setWindowProperty("Multi Camera Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        cv2.imshow("Multi Camera Detection", combined)

        # Keyboard handling (hotkeys untuk zoom & exit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            ZOOM_FACTOR = min(MAX_ZOOM, round((ZOOM_FACTOR + ZOOM_STEP) * 10) / 10.0)
            print(f"[INFO] Zoom set to {ZOOM_FACTOR:.1f}x")
        elif key == ord('x'):
            ZOOM_FACTOR = max(MIN_ZOOM, round((ZOOM_FACTOR - ZOOM_STEP) * 10) / 10.0)
            print(f"[INFO] Zoom set to {ZOOM_FACTOR:.1f}x")
        elif key == ord('r'):
            ZOOM_FACTOR = 1.0
            print("[INFO] Zoo m reset to 1.0x")
    # end while


if __name__ == "__main__":
    # Start API in background
    start_api_in_thread()

    # Run main GUI loop in main thread
    try:
        main_loop()
    finally:
        # cleanup
        for cap in caps:
            if cap:
                cap.release()
        cv2.destroyAllWindows()
        # stop OCR worker
        ocr_queue.put(None)
        print("[INFO] Exiting")

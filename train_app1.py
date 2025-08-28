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
from threading import Thread
import queue

# ===================== KONFIGURASI =====================
screen_w, screen_h = pyautogui.size()
panel_w, panel_h = screen_w // 2, screen_h // 2
TARGET_SIZE = (panel_w, panel_h)

YOLO_MODEL_PATH = "best.pt"
DETECTION_CLASSES = [2]  # kelas plat nomor
TIMEOUT_SEC = 10

# Ukuran crop untuk OCR
CROP_W = int(panel_w * 0.7)
CROP_H = int(panel_h * 0.3)
OCR_SKIP = 5  # OCR dijalankan setiap 5 frame
# =======================================================


# ===================== FUNGSI PEMBANTU =====================
def open_camera(cap):
    """Buka kamera dan set resolusi target."""
    if cap is None or not cap.isOpened():
        print("[WARNING] Gagal membuka kamera")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_SIZE[1])
    time.sleep(1)
    return cap


def create_blank_frame(text="No Camera"):
    """Buat frame kosong untuk panel kamera yang mati."""
    blank = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
    cv2.putText(blank, text, (50, TARGET_SIZE[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    return blank


def rotated_crop(image, bbox):
    """Crop, rotasi, dan ubah plat nomor ke grayscale."""
    x1, y1, x2, y2 = map(int, bbox)
    crop_img = image[y1:y2, x1:x2]
    if crop_img.size == 0:
        return None

    # Ubah ke grayscale lebih awal
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray  # kembalikan grayscale jika kontur tidak ditemukan

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect).astype(np.int32)

    width, height = int(rect[1][0]), int(rect[1][1])
    if width == 0 or height == 0:
        return gray

    # Transformasi perspektif langsung ke grayscale
    src_pts = box.astype("float32")
    dst_pts = np.array([
        [0, height - 1],
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(gray, M, (width, height))

    # Pastikan lebar > tinggi
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped  # hasil grayscale


VALID_REGION_CODES = {
    "A","B","BA","BB","BD","BE","BG","BH","BK","BL","BM","BN","BP","D","DA",
    "DB","DC","DD","DE","DG","DH","DK","DL","DM","DN","DR","DS","DT","DU",
    "E","EB","ED","F","G","H","K","KB","KH","KT","KU","L","M","N","P","R",
    "S","T","V","W","Z"
}


def filter_plate_text(ocr_result):
    """Filter hasil OCR supaya sesuai format plat Indonesia."""
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
print(f"[INFO] Menggunakan device: {device}")

model = YOLO(YOLO_MODEL_PATH).to(device)
reader = easyocr.Reader(['en'], gpu=(device == "cuda"))

camera_sources = [rtsp_stream("192.168.1.18", "101")]
caps = [open_camera(cap) for cap in camera_sources]
last_frame_times = [time.time()] * len(camera_sources)
camera_active = [cap is not None for cap in caps]

fps_history = collections.deque(maxlen=30)
prev_fps_time = time.time()

ocr_queue = queue.Queue()
ocr_result_cache = ""
frame_count = 0


def ocr_worker():
    """Thread untuk menjalankan OCR pada hasil crop grayscale."""
    global ocr_result_cache
    while True:
        crop_img = ocr_queue.get()
        if crop_img is None:
            break

        # Perbesar agar OCR lebih akurat
        processed = cv2.resize(crop_img, (0, 0), fx=2.0, fy=2.0)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        _, processed = cv2.threshold(processed, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result = reader.readtext(processed)

        if result:
            ocr_result_cache = filter_plate_text(result)


# Jalankan OCR worker
Thread(target=ocr_worker, daemon=True).start()
# =======================================================


# ===================== LOOP UTAMA =====================
while True:
    frames = []
    current_time = time.time()
    frame_count += 1

    for idx, cap in enumerate(caps):
        # Reconnect jika kamera mati
        if not camera_active[idx]:
            try:
                cap = rtsp_stream("192.168.1.18", "101")
                cap = open_camera(cap)
                caps[idx] = cap
                camera_active[idx] = True
                last_frame_times[idx] = current_time
            except ConnectionError:
                frames.append(create_blank_frame(f"Camera {idx+1}: No Camera"))
                continue

        # Baca frame
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Kamera {idx+1} tidak mengirim frame.")
            camera_active[idx] = False
            if cap:
                cap.release()
            caps[idx] = None
            frames.append(create_blank_frame(f"Camera {idx+1}: No Camera"))
            continue

        last_frame_times[idx] = current_time

        # YOLO inference
        results_gen = model(frame, classes=DETECTION_CLASSES,
                            device=device, stream=True)
        result = next(results_gen)
        annotated = result.plot()

        # Crop & OCR plat
        slot_crop = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
        cv2.putText(slot_crop, "No Plate", (10, CROP_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        if len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls.cpu().item())
                if cls in DETECTION_CLASSES:
                    coords = box.xyxy[0].cpu().numpy()
                    crop_plate = rotated_crop(frame, coords)

                    if crop_plate is not None:
                        # Konversi grayscale -> BGR hanya untuk ditampilkan
                        slot_crop = cv2.resize(cv2.cvtColor(crop_plate, cv2.COLOR_GRAY2BGR),
                                               (CROP_W, CROP_H))

                        # Kirim grayscale langsung ke OCR tiap OCR_SKIP frame
                        if frame_count % OCR_SKIP == 0:
                            ocr_queue.put(crop_plate)
                    break

        # Tempel hasil crop ke frame annotated
        h, w = annotated.shape[:2]
        y1, y2 = h - CROP_H - 10, h - 10
        x1, x2 = w - CROP_W - 10, w - 10
        annotated[y1:y2, x1:x2] = slot_crop

        # Tampilkan hasil OCR
        if ocr_result_cache:
            cv2.putText(annotated, f"Plate: {ocr_result_cache}",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(annotated, f"Camera {idx+1}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

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
    prev_fps_time = now
    fps = 1 / dt if dt > 0 else 0
    fps_history.append(fps)
    fps_avg = sum(fps_history) / len(fps_history)

    # Tampilkan hasil
    cv2.namedWindow("Multi Camera Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Multi Camera Detection",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Multi Camera Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ===================== RELEASE =====================
for cap in caps:
    if cap:
        cap.release()
cv2.destroyAllWindows()
ocr_queue.put(None)

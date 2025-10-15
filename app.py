# ==============================================================================
# ============================ START IMPORT PUSTAKA ============================
# ==============================================================================
from flask import Flask, Response, render_template_string, jsonify, abort       # Framework web Python untuk membuat API dan halaman streaming
import cv2                                                                      # Pustaka utama untuk pengolahan gambar dan video
import threading                                                                # Agar setiap kamera bisa diproses secara paralel (multithread)
import time                                                                     # Untuk menghitung waktu, delay, dan FPS
import numpy as np                                                              # Pustaka array numerik (digunakan oleh OpenCV untuk frame)
import re                                                                       # Untuk pencocokan dan pembersihan teks (misalnya hasil OCR)
from ultralytics import YOLO                                                    # Model deteksi objek untuk mendeteksi plat nomor dari gambar/video
from paddleocr import PaddleOCR                                                 # Pustaka OCR untuk membaca teks dari gambar plat
import logging                                                                  # Untuk mengatur dan membatasi pesan log di terminal
import atexit                                                                   # Agar bisa menutup kamera / menghentikan thread dengan aman saat program berhenti
# ==============================================================================
# ============================= END IMPORT PUSTAKA =============================
# ==============================================================================


# =======================================================
# ======= START NONAKTIFKAN LOG FLASK & PADDLEOCR =======
# =======================================================
log = logging.getLogger('werkzeug')                      # Mengambil logger bawaan Flask (werkzeug) agar bisa diatur level log-nya
log.setLevel(logging.ERROR)                              # Hanya tampilkan pesan error saja (tidak menampilkan GET request di terminal)
logging.getLogger("ppocr").setLevel(logging.ERROR)       # Nonaktifkan log dari PaddleOCR agar tidak terlalu ramai di terminal
# =======================================================
# ======== END NONAKTIFKAN LOG FLASK & PADDLEOCR ========
# =======================================================


app = Flask(__name__) # Membuat instance aplikasi Flask (server utama)


# =============================================================================
# ====================== START KONFIGURASI KAMERA & YOLO ======================
# =============================================================================
camera_urls = {                                                                # Daftar URL kamera CCTV (RTSP stream), ganti dengan username, password, dan IP kamera sesuai dengan kamera
    1: "rtsp://admin:itinl123@192.168.1.18:554/Streaming/Channels/101",       
    2: "rtsp://admin:itinl2025@192.168.1.64:554/Streaming/Channels/101",
    3: "rtsp://admin:itinl2025@192.168.1.91:554/Streaming/Channels/101",
    4: "rtsp://admin:itinl123@192.168.1.40:554/Streaming/Channels/101",
}
# =========================== PENGATURAN YOLO & OCR ===========================
ZOOM_FACTOR = 2.8                                                             # Faktor zoom digital (1.0 berarti tanpa zoom)
YOLO_MODEL_PATH = "best.pt"                                                    # Lokasi file model YOLO yang digunakan untuk deteksi plat nomor
YOLO_DEVICE = "cuda"                                                           # Perangkat untuk YOLO: "cuda" (GPU) atau "cpu"
YOLO_IMGSZ = 512                                                               # Ukuran gambar input untuk YOLO (semakin kecil = semakin cepat)
# ============================ PENYESUAIAN KINERJA ============================
YOLO_SKIP = 2                                                                  # Jalankan YOLO setiap N frame (misalnya 2 berarti 1 kali tiap 2 frame)
OCR_INTERVAL = 5                                                               # Jalankan OCR setiap M kali YOLO (menghemat waktu dan GPU)
GRAB_SLEEP = 0.02                                                              # Jeda antar frame saat pengambilan video (detik)
RECONNECT_DELAY = 3                                                            # Waktu tunggu (detik) sebelum mencoba koneksi ulang jika kamera terputus
# =============================================================================
# ======================= END KONFIGURASI KAMERA & YOLO =======================
# =============================================================================


# =======================================================================================
# ============================ START INISIALISASI YOLO & OCR ============================
# =======================================================================================
yolo_model = YOLO(YOLO_MODEL_PATH)                                                       # Muat model YOLO untuk deteksi plat nomor kendaraan
if YOLO_DEVICE == "cuda":                                                                # Pindahkan YOLO ke GPU jika tersedia, jika gagal pakai CPU
    try:
        yolo_model.to("cuda")
    except Exception as e:
        print("[PERINGATAN] Tidak bisa memindahkan YOLO ke CUDA, maka paksa dengan CPU:", e)       
        yolo_model.to("cpu")
# Inisialisasi PaddleOCR untuk membaca teks dari gambar plat nomor                                                                                      
ocr = PaddleOCR(use_textline_orientation=True, lang='en')                               # use_textline_orientation=True membantu saat teks miring atau miring ke samping
inference_lock = threading.Lock()                                                       # Membuat “lock” agar pemanggilan model YOLO dan OCR tidak dilakukan bersamaa (mencegah crash atau konflik penggunaan GPU)
# =======================================================================================
# ============================= END INISIALISASI YOLO & OCR =============================
# =======================================================================================


# Dictionary untuk menyimpan status dan data dari masing-masing kamera
cams = {}
for cam_id, url in camera_urls.items():
    cams[cam_id] = {
        "url": url,                 # Alamat RTSP dari kamera
        "cap": None,                # Objek VideoCapture (OpenCV)
        "latest_frame": None,       # Frame mentah terakhir dari kamera (BGR)
        "annotated_frame": None,    # Frame hasil deteksi YOLO + teks OCR
        "latest_plate": "",         # Hasil plat nomor yang sudah dinormalisasi
        "latest_raw": "",           # Hasil mentah dari OCR sebelum dibersihkan
        "lock": threading.Lock(),   # Pengunci thread agar data kamera tidak tumpang tindih
        "running": True,            # Status kamera (True = aktif, False = berhenti)
        "frame_counter": 0,         # Jumlah frame yang sudah diambil
        "yolo_counter": 0,          # Penghitung untuk interval YOLO
        "yolo_runs": 0,             # Jumlah total YOLO yang sudah dijalankan
        "fps": 0.0                  # FPS (Frame per Second) kamera tersebut
    }


# ==============================================================================
# ================= START DAFTAR PREFIX PLAT NOMOR INDONESIA ===================
# ==============================================================================
VALID_PREFIX = {
    "A","AA","AB","AD","AE","AG","B","BA","BB","BD","BE","BG","BH","BK",        # Daftar awalan plat nomor kendaraan resmi Indonesia (kode wilayah)
    "BL","BM","BN","BP","CC","CD","DA","DB","DC","DD","DE","DF","DG","DH",      # Digunakan untuk memvalidasi hasil OCR agar hanya plat Indonesia yang diterima
    "DK","DL","DM","DN","DR","DS","DT","DU","EA","EB","ED","EF","G","H",
    "K","KB","KH","KT","KU","L","M","N","P","R","S","T","W","Z"
}
# ==============================================================================
# ================= START DAFTAR PREFIX PLAT NOMOR INDONESIA ===================
# ==============================================================================
# Bersihkan hasil OCR dan ambil hanya plat utama Indonesia (tanpa tahun/pajak)
# Abaikan hasil kalau cuma angka (seperti '0923' atau '2024').
def normalize_plate(raw_text: str) -> str:
    if not raw_text:
        return ""
    # Pertama bersihkan karakter non-alfanumerik
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    if not cleaned or len(cleaned) < 3:
        return ""

    # Kedua cegah hasil tahun-only (misal 0923, 2025, 1124)
    if re.fullmatch(r'\d{2,4}', cleaned):
        return ""

    # Ketiga cari prefix sah di dalam teks
    prefix = ""
    rest = ""
    for i in range(len(cleaned)):
        for plen in (2, 1):                                                     # coba 2 huruf dulu, baru 1
            if i + plen <= len(cleaned):
                candidate = cleaned[i:i + plen]
                if candidate in VALID_PREFIX:
                    prefix = candidate
                    rest = cleaned[i + plen:]
                    break
        if prefix:
            break

    # kalau tidak ketemu prefix valid → bukan plat
    if not prefix:
        return ""
    # Keempat pisahkan nomor dan huruf belakang
    m = re.match(r'(\d+)([A-Z0-9]*)', rest)
    if not m:
        return ""

    number = m.group(1)
    suffix = m.group(2)

    # Kelima hapus indikasi tahun/pajak di akhir (contoh: 0923, 09.24, 2024)
    suffix = re.sub(r'(\d{2,4})$', '', suffix)

    # Keenam gabungkan hasil akhir
    plate = f"{prefix} {number}"
    if suffix:
        ssplit = re.findall(r'[A-Z]+', suffix)                                  # hanya huruf belakang
        if ssplit:
            plate += " " + " ".join(ssplit)

    # Ketujuh validasi panjang plat minimal
    if len(plate.replace(" ", "")) < 5:
        return ""

    return plate.strip()
# ==============================================================================
# ================== END DAFTAR PREFIX PLAT NOMOR INDONESIA ====================
# ==============================================================================

# ---------- utility: no-signal image ----------
def no_signal_frame(w=640, h=480):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    text = "NO SIGNAL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h // 2) + (th // 2)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return img

# ---------- camera open with retries ----------
def open_cam(url):
    # pakai FFmpeg backend + non-buffer
    cap = cv2.VideoCapture(
        url + "?rtsp_transport=tcp&fflags=nobuffer&flags=low_delay",
        cv2.CAP_FFMPEG
    )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # buang buffer internal

    time.sleep(0.3)
    if cap.isOpened():
        return cap
    else:
        try:
            cap.release()
        except:
            pass
        return None

# ---------- grabber thread per camera ----------
def cam_grabber(cam_id):
    cam = cams[cam_id]
    while cam["running"]:
        if cam["cap"] is None or not cam["cap"].isOpened():
            print(f"[CAM {cam_id}] opening {cam['url']} ...")
            newcap = open_cam(cam["url"])
            if newcap is None:
                print(f"[CAM {cam_id}] failed to open, retry in {RECONNECT_DELAY}s")
                time.sleep(RECONNECT_DELAY)
                continue
            cam["cap"] = newcap
            print(f"[CAM {cam_id}] opened")

        try:
            for _ in range(3):
                cam["cap"].grab()
            ret, frame = cam["cap"].read()
        except Exception as ex:
            print(f"[CAM {cam_id}] read exception: {ex}")
            ret = False
            frame = None

        if not ret or frame is None:
            # connection issue -> release and try reconnect
            print(f"[CAM {cam_id}] frame read failed, releasing and reconnecting")
            try:
                cam["cap"].release()
            except:
                pass
            cam["cap"] = None
            # mark no signal
            with cam["lock"]:
                cam["latest_frame"] = None
            time.sleep(1)
            continue

        # compute FPS approx per second
        cam["frame_counter"] += 1
        # use a simple per-camera sliding window: recompute every 1s
        now = time.time()
        if "fps_ts" not in cam:
            cam["fps_ts"] = now
            cam["fps_count"] = 0

        cam["fps_count"] += 1
        if now - cam["fps_ts"] >= 1.0:
            with cam["lock"]:
                cam["fps"] = cam["fps_count"] / (now - cam["fps_ts"])
            cam["fps_count"] = 0
            cam["fps_ts"] = now

        # store latest frame (overwrite)
        with cam["lock"]:
            cam["latest_frame"] = frame

        time.sleep(GRAB_SLEEP)

# ---------- detector thread per camera ----------
def safe_extract_text_from_ocr(ocr_result):
    if not ocr_result:
        return ""

    texts_with_pos = []

    try:
        # PaddleOCR result → [ [ [box, (text, conf)], ... ] ]
        for chunk in ocr_result:
            if isinstance(chunk, list):
                for item in chunk:
                    try:
                        box = item[0]
                        txt = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                        if isinstance(txt, str) and txt.strip():
                            # Ambil posisi X paling kiri (rata-rata dari dua titik kiri)
                            x_pos = (box[0][0] + box[3][0]) / 2.0 if isinstance(box, list) and len(box) >= 4 else 0
                            texts_with_pos.append((x_pos, txt.strip()))
                    except Exception:
                        continue
            else:
                try:
                    box = chunk[0]
                    txt = chunk[1][0] if isinstance(chunk[1], (list, tuple)) else chunk[1]
                    if isinstance(txt, str) and txt.strip():
                        x_pos = (box[0][0] + box[3][0]) / 2.0 if isinstance(box, list) and len(box) >= 4 else 0
                        texts_with_pos.append((x_pos, txt.strip()))
                except Exception:
                    continue
    except Exception:
        pass

    # Urutkan dari kiri ke kanan (X paling kecil dulu)
    texts_sorted = [t for _, t in sorted(texts_with_pos, key=lambda x: x[0])]
    return " ".join(texts_sorted).strip()

def cam_detector(cam_id):
    cam = cams[cam_id]
    yolo_runs = 0

    while cam["running"]:
        with cam["lock"]:
            frame = cam["latest_frame"].copy() if cam["latest_frame"] is not None else None

        if frame is None:
            # set annotated to no-signal for viewers
            with cam["lock"]:
                cam["annotated_frame"] = no_signal_frame()
            time.sleep(0.1)
            continue

        # apply zoom safely
        try:
            h, w = frame.shape[:2]
            if ZOOM_FACTOR > 1.0:
                new_w = int(w / ZOOM_FACTOR)
                new_h = int(h / ZOOM_FACTOR)
                x1 = (w - new_w) // 2
                y1 = (h - new_h) // 2
                x2 = x1 + new_w
                y2 = y1 + new_h
                cropped = frame[y1:y2, x1:x2]
                proc_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                proc_frame = frame
        except Exception as e:
            proc_frame = frame

        cam["yolo_counter"] += 1
        do_yolo = (cam["yolo_counter"] % YOLO_SKIP == 0)

        annotated = proc_frame.copy()

        if do_yolo:
            # run YOLO under inference lock to avoid concurrent CUDA calls
            with inference_lock:
                try:
                    results = yolo_model(proc_frame, verbose=False, imgsz=YOLO_IMGSZ)
                except Exception as e:
                    print(f"[CAM {cam_id}] YOLO exception: {e}")
                    results = None

            if results is not None and len(results) > 0:
                # Ultralytics returns list-like results; we inspect first
                try:
                    boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0], "boxes") and len(results[0].boxes) > 0 else []
                except Exception:
                    # fallback: empty
                    boxes = []

                # draw boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # OCR sampling: every OCR_INTERVAL yolo runs do OCR on first box(s)
                yolo_runs += 1
                do_ocr = (yolo_runs % OCR_INTERVAL == 0)

                if do_ocr and len(boxes) > 0:
                    # prefer biggest box (likely plate)
                    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
                    # try OCR on up to 2 biggest boxes
                    for i_box, box in enumerate(boxes_sorted[:2]):
                        x1, y1, x2, y2 = map(int, box[:4])
                        # padding a little
                        px = max(0, x1 - 5); py = max(0, y1 - 5)
                        px2 = min(proc_frame.shape[1], x2 + 5); py2 = min(proc_frame.shape[0], y2 + 5)
                        plate_crop = proc_frame[py:py2, px:px2]
                        if plate_crop is None or plate_crop.size == 0:
                            continue

                        try:
                            # call OCR (safe)
                            ocr_result = ocr.ocr(plate_crop)
                        except Exception as e:
                            print(f"[CAM {cam_id}] OCR error: {e}")
                            ocr_result = None

                        raw_text = safe_extract_text_from_ocr(ocr_result)
                        normalized = normalize_plate(raw_text) if raw_text else ""

                        # update latest plate if non-empty
                        if normalized:
                            with cam["lock"]:
                                cam["latest_plate"] = normalized
                                cam["latest_raw"] = raw_text
                                cam["last_plate_ts"] = time.time()
                            # draw small text near box
                            cv2.putText(annotated, normalized, (x1, max(20, y1-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                            # break after first valid OCR
                            break

        # overlay FPS (from grabber)
        with cam["lock"]:
            fps_val = cam.get("fps", 0.0)
        # try:
        #     cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # except Exception:
        #     pass

        # set annotated_frame so stream can use it
        with cam["lock"]:
            cam["annotated_frame"] = annotated

        # small sleep to yield
        time.sleep(0.01)

# ---------- start threads ----------
for cam_id in cams.keys():
    t_grab = threading.Thread(target=cam_grabber, args=(cam_id,), daemon=True)
    t_grab.start()
    t_det = threading.Thread(target=cam_detector, args=(cam_id,), daemon=True)
    t_det.start()
    print(f"[MAIN] started threads for cam {cam_id}")

# ---------- Flask endpoints ----------
# index (web UI with grid)
index_html = """
<!doctype html>
<html lang="en">

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multi-Cam Plate</title>
  <link rel="icon" href="data:," />
  <style>
    body {
      background: #88000b;
      color: #eee;
      font-family: Segoe UI, Tahoma, Verdana;
      margin: 0;
      padding: 10px;
    }

    h1 {
      color: #ffffff;
      margin-bottom: 10px;
      text-align: center;
      margin-top: 1px;
    }

    h4 {
      color: #ffffff;
      margin-bottom: 14px;
      text-align: center;
      margin-top: 1px;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      max-width: 1200px;
      margin: 0 auto;
    }

    .cam {
      background: #222;
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.6);
    }

    .video-wrapper {
      position: relative;
      padding-bottom: 56.25%;
      height: 0;
      overflow: hidden;
      border-radius: 6px;
      background: #000;
      margin-bottom: 10px;
    }

    img {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    code {
      background: #88000b;
      color: #ffffff;
      padding: 4px 8px;
      border-radius: 4px;
      font-weight: bold;
      display: inline-block;
      min-width: 80px;
      text-align: center;
    }

    .status-dot {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      margin-right: 8px;
      vertical-align: middle;
      background: #999;
    }

    @media (max-width:650px) {
      .grid {
        grid-template-columns: 1fr;
      }

      body {
        padding: 10px;
      }
    }
  </style>
</head>

<body>
  <h1>Multi-Camera Plate Monitor</h1>

  {% if cams %}
  <div class="grid">
    {% for cam_id in cams %}
    <div class="cam">
      <h4>Cam {{ cam_id }}</h4>
      <div class="video-wrapper">
        <a href="/video/{{ cam_id }}" target="_blank">
          <img src="/video/{{ cam_id }}?t={{ loop.index }}" alt="cam{{ cam_id }}">
        </a>
      </div>
      <p>
        <span class="status-dot" id="dot{{ cam_id }}"></span>
        <code id="p{{ cam_id }}">loading...</code>
        &nbsp;
        <code id="f{{ cam_id }}">-</code>
      </p>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <p style="text-align:center;">No camera streams available.</p>
  {% endif %}

  <script>
    const camIds = {{ cams | safe }};
    const COLOR_ONLINE = '#2ecc71';   // Hijau = aktif
    const COLOR_LAGGING = '#f1c40f';  // Kuning = lemot
    const COLOR_OFFLINE = '#e74c3c';  // Merah = mati
    const COLOR_UNKNOWN = '#999';     // Abu = error

    // Update tiap 500ms
    setInterval(() => {
      camIds.forEach(id => {
        fetch(`/plate/${id}`)
          .then(r => r.json())
          .then(j => {
            const plate = document.getElementById('p' + id);
            const fps = document.getElementById('f' + id);
            const dot = document.getElementById('dot' + id);

            const currentFPS = (j.fps !== undefined) ? j.fps : -1;
            const isOnline = j.status === 'online';

            plate.innerText = j.plate || 'Waiting...';
            fps.innerText = (currentFPS >= 0) ? currentFPS.toFixed(1) : '-';

            if (isOnline) {
              if (currentFPS < 4.5) dot.style.background = COLOR_LAGGING;
              else dot.style.background = COLOR_ONLINE;
            } else {
              dot.style.background = COLOR_OFFLINE;
            }
          })
          .catch(() => {
            document.getElementById('p' + id).innerText = '(error)';
            document.getElementById('f' + id).innerText = '-';
            document.getElementById('dot' + id).style.background = COLOR_UNKNOWN;
          });
      });
    }, 500);
  </script>
</body>

</html>
"""

@app.route('/')
def index():
    return render_template_string(index_html, cams=list(cams.keys()))

def mjpeg_response_generator(cam_id):
    cam = cams.get(cam_id)
    if cam is None:
        return

    while True:
        with cam["lock"]:
            frame = cam["annotated_frame"].copy() if cam["annotated_frame"] is not None else None

        if frame is None:
            frame = no_signal_frame()

        # encode
        try:
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret:
                frame = no_signal_frame()
                ret, buf = cv2.imencode('.jpg', frame)
        except Exception:
            frame = no_signal_frame()
            ret, buf = cv2.imencode('.jpg', frame)

        jpg = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.05)  # control approx frame rate for clients

@app.route('/video/<int:cam_id>')
def video(cam_id):
    if cam_id not in cams:
        abort(404)
    return Response(
    mjpeg_response_generator(cam_id),
    mimetype='multipart/x-mixed-replace; boundary=frame',
    headers={'Cache-Control': 'no-store, must-revalidate'}
    )

@app.route('/plate/<int:cam_id>')
def plate_api(cam_id):
    cam = cams.get(cam_id)
    if cam is None:
        abort(404)

    with cam["lock"]:
        # cek timeout plat (reset setelah 5 detik tidak update)
        if cam["latest_plate"] and (time.time() - cam.get("last_plate_ts", 0) > 5):
            cam["latest_plate"] = ""
            cam["latest_raw"] = ""

        # Tentukan status kamera
        if cam["latest_frame"] is None:
            status = "offline"
        else:
            # Jika FPS rendah banget dianggap lagging
            if cam.get("fps", 0) < 1.0:
                status = "lagging"
            else:
                status = "online"

        return jsonify({
            "plate": cam["latest_plate"],
            "raw": cam["latest_raw"],
            "fps": float(cam.get("fps", 0.0)),
            "status": status
        })
# ---------- graceful shutdown ----------
def stop_all():
    for cam in cams.values():
        cam["running"] = False
        try:
            if cam["cap"] is not None:
                cam["cap"].release()
        except:
            pass

atexit.register(stop_all)

# ---------- run server ----------
if __name__ == "__main__":
    # Run Flask. For production, use a WSGI server.
    app.run(host="0.0.0.0", port=5001, threaded=True)
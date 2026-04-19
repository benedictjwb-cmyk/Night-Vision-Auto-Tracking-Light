#!/usr/bin/env python3
"""
Single-file tracker + light + servos + MJPEG phone stream + realtime description overlay
+ smoothness logging (angles, velocities, accelerations) to CSV.

Phone view:
  http://<JETSON_IP>:5000/

Outputs:
  zigzag_smoothness.csv  (t, pan/tilt deg, pan/tilt vel deg/s, pan/tilt acc deg/s^2, plus detection + cx/cy)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import time
import csv
import textwrap
import threading
import time as _time

from ultralytics import YOLO
import Jetson.GPIO as GPIO
from flask import Flask, Response

# =========================================================
# STREAMING (MJPEG)
# =========================================================
app = Flask(__name__)
_latest_jpeg = None
_jpeg_lock = threading.Lock()

def _set_latest_frame(bgr_frame, jpeg_quality=80):
    """Encode and store the latest frame for MJPEG streaming."""
    global _latest_jpeg
    ok, jpg = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return
    with _jpeg_lock:
        _latest_jpeg = jpg.tobytes()

def _mjpeg_generator():
    """Yields multipart JPEG frames."""
    global _latest_jpeg
    while True:
        with _jpeg_lock:
            frame = _latest_jpeg
        if frame is None:
            _time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        _time.sleep(0.02)  # throttle a bit

@app.route("/")
def stream():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def start_stream_server(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, threaded=True, use_reloader=False)

# =========================================================
# REALTIME DESCRIPTION (people only because classes=0)
# =========================================================
def describe_frame_yolo(results0, w, h, top_k=3):
    boxes = results0.boxes
    n = len(boxes)
    if n == 0:
        return "I can’t see a person in frame."

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    order = np.argsort(-conf)[:min(top_k, len(conf))]

    def where(cx, cy):
        horiz = "left" if cx < w/3 else ("right" if cx > 2*w/3 else "center")
        vert  = "top"  if cy < h/3 else ("bottom" if cy > 2*h/3 else "middle")
        return horiz, vert

    parts = []
    for j, idx in enumerate(order):
        x1, y1, x2, y2 = xyxy[idx]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        area = (x2 - x1) * (y2 - y1)
        area_frac = float(area) / float(w * h)

        horiz, vert = where(cx, cy)

        if area_frac > 0.20:
            size_txt = "very close"
        elif area_frac > 0.08:
            size_txt = "close"
        elif area_frac > 0.03:
            size_txt = "mid-distance"
        else:
            size_txt = "far"

        parts.append(f"person {j+1} is {vert}-{horiz}, {size_txt} (conf {conf[idx]:.2f})")

    if n == 1:
        return "I see 1 person: " + parts[0] + "."
    return f"I see {n} people: " + "; ".join(parts) + "."

def overlay_description(frame_bgr, desc, w):
    lines = textwrap.wrap(desc, width=60)[:2]  # 2 lines max
    pad = 8
    line_h = 26
    box_h = pad * 2 + line_h * len(lines)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (5, 5), (w - 5, 5 + box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)

    y = 5 + pad + 18
    for line in lines:
        cv2.putText(out, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h
    return out

# =========================================================
# LIGHT (GPIO) CONFIG
# =========================================================
LIGHT_PIN = 12
LIGHT_HOLD_S = 5.0

def unlock_light_pin():
    os.system("sudo busybox devmem 0x02434088 32 0x400")

def init_light_gpio():
    GPIO.setwarnings(False)
    try:
        GPIO.cleanup()
    except Exception:
        pass
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LIGHT_PIN, GPIO.OUT, initial=GPIO.LOW)

def set_light(on: bool):
    GPIO.output(LIGHT_PIN, GPIO.HIGH if on else GPIO.LOW)

# =========================================================
# PCA9685 HELPERS
# =========================================================
def us_to_ticks(pulse_us: float, freq_hz: float) -> int:
    period_us = 1_000_000.0 / float(freq_hz)
    ticks = int((float(pulse_us) / period_us) * 4096.0)
    return max(0, min(4095, ticks))

def set_pwm_us(pca, channel: int, pulse_us: float):
    ticks = us_to_ticks(pulse_us, pca.frequency)
    pca.channels[channel].duty_cycle = ticks << 4

def set_servo_angle(pca, channel: int, angle: float,
                    angle_max: float = 270.0,
                    pulse_min_us: float = 500.0,
                    pulse_max_us: float = 2500.0):
    a = max(0.0, min(float(angle_max), float(angle)))
    pulse = pulse_min_us + (a / float(angle_max)) * (pulse_max_us - pulse_min_us)
    set_pwm_us(pca, channel, pulse)

def init_pca9685(freq_hz=50):
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    print("Initializing PCA9685...")
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = freq_hz
    return pca

# =========================================================
# PAN/TILT CONFIG
# =========================================================
PAN_CH  = 0
TILT_CH = 1
PAN_MIN,  PAN_MAX  = 0.0, 270.0
TILT_MIN, TILT_MAX = 0.0, 270.0

pan_angle  = (PAN_MIN + PAN_MAX) / 2
tilt_angle = (TILT_MIN + TILT_MAX) / 2

# Control params (keep as you like)
DEADZONE_PX = 80
KP_PAN  = 15.0
KP_TILT = 15.0

# If you want less jerk, reduce these (e.g., 0.7)
MAX_STEP_PAN_DEG  = 1.2
MAX_STEP_TILT_DEG = 1.2

PAN_SIGN  = -1.0
TILT_SIGN = 1.0

# =========================================================
# LOGGING (angles + velocities + accelerations)
# =========================================================
LOG_PATH = "zigzag_smoothness.csv"

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =========================================================
# LOAD YOLO + INIT HW
# =========================================================
print("Loading model...")
model = YOLO("yolov8n.pt")
model.to("cuda")

pca = init_pca9685(freq_hz=50)
set_servo_angle(pca, PAN_CH,  pan_angle)
set_servo_angle(pca, TILT_CH, tilt_angle)

unlock_light_pin()
init_light_gpio()
set_light(False)
last_human_time = 0.0
light_on = False

gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! "
    "appsink max-buffers=1 drop=true sync=false"
)

cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

# Start stream server
threading.Thread(target=start_stream_server, daemon=True).start()
print("📡 Stream running. Open on phone: http://<JETSON_IP>:5000/")

# Open CSV
log_f = open(LOG_PATH, "w", newline="")
log = csv.writer(log_f)
log.writerow([
    "t",
    "pan_deg", "tilt_deg",
    "pan_vel_deg_s", "tilt_vel_deg_s",
    "pan_acc_deg_s2", "tilt_acc_deg_s2",
    "human", "cx", "cy"
])

t0 = time.time()
prev_t = None

# Derivative state
prev_pan = None
prev_tilt = None
prev_pan_vel = 0.0
prev_tilt_vel = 0.0

# Simple FPS estimate for overlay/debug
fps = 0.0
last_fps_t = time.time()

# =========================================================
# MAIN LOOP
# =========================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        now = time.time()
        t = now - t0

        h, w = frame.shape[:2]
        cx_img, cy_img = w // 2, h // 2

        results = model(frame, classes=0, imgsz=320, verbose=False)
        human_detected = len(results[0].boxes) > 0

        # ========= LIGHT LOGIC (with hold) =========
        if human_detected:
            last_human_time = now

        should_be_on = (now - last_human_time) < LIGHT_HOLD_S

        if should_be_on != light_on:
            light_on = should_be_on
            set_light(light_on)
            print("💡 LIGHT ON" if light_on else "⬇️ LIGHT OFF")
        # ===========================================

        cx = cy = None

        # ========= SERVO CONTROL =========
        if human_detected:
            boxes = results[0].boxes.xyxy.detach().cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int(np.argmax(areas))
            x1, y1, x2, y2 = boxes[i]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            diff_x = cx - cx_img
            diff_y = cy - cy_img

            # PAN
            if abs(diff_x) > DEADZONE_PX:
                err_x = diff_x / float(cx_img)
                step = KP_PAN * err_x
                if abs(step) > 0.5:
                    step = clamp(step, -MAX_STEP_PAN_DEG, MAX_STEP_PAN_DEG)
                    pan_angle += PAN_SIGN * step
                    pan_angle = clamp(pan_angle, PAN_MIN, PAN_MAX)
                    set_servo_angle(pca, PAN_CH, pan_angle)

            # TILT
            if abs(diff_y) > DEADZONE_PX:
                err_y = diff_y / float(cy_img)
                step = KP_TILT * err_y
                if abs(step) > 0.5:
                    step = clamp(step, -MAX_STEP_TILT_DEG, MAX_STEP_TILT_DEG)
                    tilt_angle += TILT_SIGN * step
                    tilt_angle = clamp(tilt_angle, TILT_MIN, TILT_MAX)
                    set_servo_angle(pca, TILT_CH, tilt_angle)
        # ===========================================

        # ========= DERIVATIVES (vel/acc) =========
        pan_vel = tilt_vel = 0.0
        pan_acc = tilt_acc = 0.0

        if prev_t is not None:
            dt = t - prev_t
            if dt > 1e-6:
                if prev_pan is None:
                    prev_pan = pan_angle
                    prev_tilt = tilt_angle

                pan_vel = (pan_angle - prev_pan) / dt
                tilt_vel = (tilt_angle - prev_tilt) / dt

                pan_acc = (pan_vel - prev_pan_vel) / dt
                tilt_acc = (tilt_vel - prev_tilt_vel) / dt

                prev_pan_vel = pan_vel
                prev_tilt_vel = tilt_vel

                prev_pan = pan_angle
                prev_tilt = tilt_angle

        prev_t = t
        # ===========================================

        # ========= LOG ROW =========
        log.writerow([
            f"{t:.4f}",
            f"{pan_angle:.3f}", f"{tilt_angle:.3f}",
            f"{pan_vel:.3f}", f"{tilt_vel:.3f}",
            f"{pan_acc:.3f}", f"{tilt_acc:.3f}",
            1 if human_detected else 0,
            "" if cx is None else cx,
            "" if cy is None else cy,
        ])

        # Flush occasionally so you don't lose data if you Ctrl+C
        if int(t * 10) % 10 == 0:  # ~once per second
            log_f.flush()

        # ========= ANNOTATE + DESCRIPTION + STREAM =========
        annotated = results[0].plot()

        desc = describe_frame_yolo(results[0], w, h)
        annotated = overlay_description(annotated, desc, w)

        # Small stats overlay (optional but useful for the test)
        now_t = time.time()
        dt_fps = now_t - last_fps_t
        if dt_fps > 1e-6:
            fps = 0.9 * fps + 0.1 * (1.0 / dt_fps)
        last_fps_t = now_t

        stats = f"Light:{'ON' if light_on else 'OFF'}  FPS:{fps:.1f}  Pan:{pan_angle:.1f}  Tilt:{tilt_angle:.1f}"
        cv2.putText(annotated, stats, (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        _set_latest_frame(annotated, jpeg_quality=80)

        # Local window (optional)
        cv2.imshow("CSI Human Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    try:
        log_f.close()
        print(f"✅ Saved log to {LOG_PATH}")
    except Exception:
        pass

    try:
        set_light(False)
        GPIO.cleanup()
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()


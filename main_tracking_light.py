#!/usr/bin/env python3
"""
Single-file tracker + light + servos + MJPEG phone stream + realtime description overlay
+ adaptive deadzone based on target motion.

Phone view:
  http://<JETSON_IP>:5000/
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import time
import textwrap
import threading
import time as _time
import math

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
        _time.sleep(0.02)

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
    lines = textwrap.wrap(desc, width=60)[:2]
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

# Proportional gains
KP_PAN  = 15.0
KP_TILT = 15.0

# Motion-adaptive deadzone parameters
DEADZONE_MIN_PX = 20.0
DEADZONE_MAX_PX = 100.0
ALPHA_MOTION = 0.7
GV = 0.1 * DEADZONE_MAX_PX

# Minimum and maximum angular command per update
UMIN_DEG = 0.5
MAX_STEP_PAN_DEG  = 1.0
MAX_STEP_TILT_DEG = 1.0

PAN_SIGN  = -1.0
TILT_SIGN = 1.0

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

threading.Thread(target=start_stream_server, daemon=True).start()
print("📡 Stream running. Open on phone: http://<JETSON_IP>:5000/")

# Adaptive deadzone state
prev_target_cx = None
prev_target_cy = None
motion_filt_px = 0.0

# Simple FPS estimate
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

        h, w = frame.shape[:2]
        cx_img, cy_img = w // 2, h // 2

        results = model(frame, classes=0, imgsz=320, verbose=False)
        human_count = len(results[0].boxes)
        human_detected = human_count > 0

        # ========= LIGHT LOGIC =========
        if human_detected:
            last_human_time = now

        should_be_on = (now - last_human_time) < LIGHT_HOLD_S

        if should_be_on != light_on:
            light_on = should_be_on
            set_light(light_on)
            print("💡 LIGHT ON" if light_on else "⬇️ LIGHT OFF")

        # Defaults
        cx = cy = None
        current_deadzone_px = DEADZONE_MAX_PX

        # ========= SERVO CONTROL =========
        if human_detected:
            boxes = results[0].boxes.xyxy.detach().cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int(np.argmax(areas))
            x1, y1, x2, y2 = boxes[i]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Motion estimate from target displacement between frames
            if prev_target_cx is None or prev_target_cy is None:
                target_motion_px = 0.0
                motion_filt_px = 0.0
            else:
                dx_motion = float(cx - prev_target_cx)
                dy_motion = float(cy - prev_target_cy)
                target_motion_px = math.sqrt(dx_motion**2 + dy_motion**2)
                motion_filt_px = (
                    ALPHA_MOTION * motion_filt_px +
                    (1.0 - ALPHA_MOTION) * target_motion_px
                )

            # Adaptive deadzone
            current_deadzone_px = clamp(
                DEADZONE_MAX_PX - GV * motion_filt_px,
                DEADZONE_MIN_PX,
                DEADZONE_MAX_PX
            )

            # Save target centre for next frame
            prev_target_cx = cx
            prev_target_cy = cy

            # Tracking error relative to image centre
            diff_x = float(cx - cx_img)
            diff_y = float(cy - cy_img)

            err_x = diff_x / float(cx_img)
            err_y = diff_y / float(cy_img)

            # PAN
            if abs(diff_x) > current_deadzone_px:
                step = KP_PAN * err_x
                if abs(step) >= UMIN_DEG:
                    step = clamp(step, -MAX_STEP_PAN_DEG, MAX_STEP_PAN_DEG)
                    pan_angle += PAN_SIGN * step
                    pan_angle = clamp(pan_angle, PAN_MIN, PAN_MAX)
                    set_servo_angle(pca, PAN_CH, pan_angle)

            # TILT
            if abs(diff_y) > current_deadzone_px:
                step = KP_TILT * err_y
                if abs(step) >= UMIN_DEG:
                    step = clamp(step, -MAX_STEP_TILT_DEG, MAX_STEP_TILT_DEG)
                    tilt_angle += TILT_SIGN * step
                    tilt_angle = clamp(tilt_angle, TILT_MIN, TILT_MAX)
                    set_servo_angle(pca, TILT_CH, tilt_angle)

        else:
            # Reset motion state so reacquisition does not create a false spike
            prev_target_cx = None
            prev_target_cy = None
            motion_filt_px = 0.0

        # ========= ANNOTATE + DESCRIPTION + STREAM =========
        annotated = results[0].plot()
        desc = describe_frame_yolo(results[0], w, h)
        annotated = overlay_description(annotated, desc, w)

        now_t = time.time()
        dt_fps = now_t - last_fps_t
        if dt_fps > 1e-6:
            fps = 0.9 * fps + 0.1 * (1.0 / dt_fps)
        last_fps_t = now_t

        stats = (
            f"Light:{'ON' if light_on else 'OFF'}  "
            f"People:{human_count}  "
            f"FPS:{fps:.1f}  "
            f"Pan:{pan_angle:.1f}  Tilt:{tilt_angle:.1f}  "
            f"DZ:{current_deadzone_px:.1f}px"
        )
        cv2.putText(annotated, stats, (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        _set_latest_frame(annotated, jpeg_quality=80)

        cv2.imshow("CSI Human Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    try:
        set_light(False)
        GPIO.cleanup()
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()

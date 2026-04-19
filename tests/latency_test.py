#!/usr/bin/env python3
"""
Latency test based on the working tracking + light + stream script.

Logs:
- inference latency
- command latency
- total software latency

Outputs:
  latency_results.csv

Phone view:
  http://<JETSON_IP>:5000/
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

def _set_latest_frame(bgr_frame, jpeg_quality=70):
    global _latest_jpeg
    ok, jpg = cv2.imencode(
        ".jpg",
        bgr_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        return
    with _jpeg_lock:
        _latest_jpeg = jpg.tobytes()

def _mjpeg_generator():
    global _latest_jpeg
    while True:
        with _jpeg_lock:
            frame = _latest_jpeg
        if frame is None:
            _time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        _time.sleep(0.02)

@app.route("/")
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def start_stream_server(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, threaded=True, use_reloader=False)

# =========================================================
# REALTIME DESCRIPTION
# =========================================================
def describe_frame_yolo_from_boxes(boxes_xyxy, confs, w, h, top_k=2):
    n = len(boxes_xyxy)
    if n == 0:
        return "I can’t see a person in frame."

    order = np.argsort(-confs)[:min(top_k, len(confs))]

    def where(cx, cy):
        horiz = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
        vert  = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
        return horiz, vert

    parts = []
    for j, idx in enumerate(order):
        x1, y1, x2, y2 = boxes_xyxy[idx]
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

        parts.append(
            f"person {j+1} is {vert}-{horiz}, {size_txt} (conf {confs[idx]:.2f})"
        )

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
        cv2.putText(
            out, line, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )
        y += line_h
    return out

# =========================================================
# LIGHT (GPIO) CONFIG
# =========================================================
LIGHT_PIN = 12
LIGHT_HOLD_S = 5.0

def unlock_light_pin():
    # Run this manually before the script if sudo prompt causes issues:
    # sudo busybox devmem 0x02434088 32 0x400
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

def set_servo_angle(
    pca,
    channel: int,
    angle: float,
    angle_max: float = 270.0,
    pulse_min_us: float = 500.0,
    pulse_max_us: float = 2500.0
):
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
PAN_CH = 0
TILT_CH = 1
PAN_MIN, PAN_MAX = 0.0, 270.0
TILT_MIN, TILT_MAX = 0.0, 270.0

pan_angle = (PAN_MIN + PAN_MAX) / 2
tilt_angle = (TILT_MIN + TILT_MAX) / 2

DEADZONE_PX = 80
KP_PAN = 15.0
KP_TILT = 15.0
MAX_STEP_PAN_DEG = 1.2
MAX_STEP_TILT_DEG = 1.2

PAN_SIGN = -1.0
TILT_SIGN = 1.0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =========================================================
# LATENCY LOGGING
# =========================================================
LOG_PATH = "latency_results.csv"

# =========================================================
# LOAD YOLO + INIT HW
# =========================================================
print("Loading model...")
model = YOLO("yolov8n.pt")
model.to("cuda")

pca = init_pca9685(freq_hz=50)
set_servo_angle(pca, PAN_CH, pan_angle)
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

log_f = open(LOG_PATH, "w", newline="")
log = csv.writer(log_f)
log.writerow([
    "frame_idx",
    "t_frame_start",
    "t_after_inference",
    "t_first_command",
    "human",
    "bbox_cx", "bbox_cy",
    "diff_x", "diff_y",
    "pan_step_deg", "tilt_step_deg",
    "light_changed",
    "inference_latency_ms",
    "command_latency_ms",
    "total_software_latency_ms"
])

fps = 0.0
last_fps_t = time.time()
frame_count = 0

INFER_W = 640
INFER_H = 360

# =========================================================
# MAIN LOOP
# =========================================================
try:
    while True:
        t_frame_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame read failed.")
            break

        now = time.time()
        frame_count += 1

        h, w = frame.shape[:2]
        cx_img, cy_img = w // 2, h // 2

        small = cv2.resize(frame, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        results = model(small, classes=0, imgsz=320, verbose=False)
        t_after_inference = time.perf_counter()

        human_detected = len(results[0].boxes) > 0

        cx = cy = None
        x1 = y1 = x2 = y2 = None
        boxes = np.empty((0, 4), dtype=np.float32)
        confs = np.array([])

        if human_detected:
            boxes_small = results[0].boxes.xyxy.detach().cpu().numpy()
            confs = results[0].boxes.conf.detach().cpu().numpy()

            sx = w / float(INFER_W)
            sy = h / float(INFER_H)

            boxes = boxes_small.copy()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        # We log the first actuation command timestamp in this frame
        t_first_command = None
        light_changed = 0
        pan_step_logged = 0.0
        tilt_step_logged = 0.0
        diff_x = ""
        diff_y = ""

        # ========= LIGHT LOGIC =========
        if human_detected:
            last_human_time = now

        should_be_on = (now - last_human_time) < LIGHT_HOLD_S

        if should_be_on != light_on:
            light_on = should_be_on
            set_light(light_on)
            light_changed = 1
            if t_first_command is None:
                t_first_command = time.perf_counter()

        # ========= SERVO CONTROL =========
        if human_detected:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int(np.argmax(areas))
            x1, y1, x2, y2 = boxes[i]

            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)

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
                    pan_step_logged = step
                    if t_first_command is None:
                        t_first_command = time.perf_counter()

            # TILT
            if abs(diff_y) > DEADZONE_PX:
                err_y = diff_y / float(cy_img)
                step = KP_TILT * err_y
                if abs(step) > 0.5:
                    step = clamp(step, -MAX_STEP_TILT_DEG, MAX_STEP_TILT_DEG)
                    tilt_angle += TILT_SIGN * step
                    tilt_angle = clamp(tilt_angle, TILT_MIN, TILT_MAX)
                    set_servo_angle(pca, TILT_CH, tilt_angle)
                    tilt_step_logged = step
                    if t_first_command is None:
                        t_first_command = time.perf_counter()

        if t_first_command is None:
            t_first_command = time.perf_counter()

        inference_latency_ms = (t_after_inference - t_frame_start) * 1000.0
        command_latency_ms = (t_first_command - t_after_inference) * 1000.0
        total_software_latency_ms = (t_first_command - t_frame_start) * 1000.0

        log.writerow([
            frame_count,
            f"{t_frame_start:.6f}",
            f"{t_after_inference:.6f}",
            f"{t_first_command:.6f}",
            1 if human_detected else 0,
            "" if cx is None else cx,
            "" if cy is None else cy,
            diff_x,
            diff_y,
            f"{pan_step_logged:.4f}",
            f"{tilt_step_logged:.4f}",
            light_changed,
            f"{inference_latency_ms:.3f}",
            f"{command_latency_ms:.3f}",
            f"{total_software_latency_ms:.3f}"
        ])

        if frame_count % 10 == 0:
            log_f.flush()

        # ========= ANNOTATE + STREAM =========
        annotated = frame.copy()

        if human_detected and x1 is not None:
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        desc = describe_frame_yolo_from_boxes(boxes, confs, w, h)
        annotated = overlay_description(annotated, desc, w)

        now_t = time.time()
        dt_fps = now_t - last_fps_t
        if dt_fps > 1e-6:
            fps = 0.9 * fps + 0.1 * (1.0 / dt_fps)
        last_fps_t = now_t

        stats = (
            f"Light:{'ON' if light_on else 'OFF'}  "
            f"FPS:{fps:.1f}  "
            f"Inf:{inference_latency_ms:.1f}ms  "
            f"Cmd:{command_latency_ms:.1f}ms  "
            f"Tot:{total_software_latency_ms:.1f}ms"
        )
        cv2.putText(
            annotated, stats, (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        if frame_count % 2 == 0:
            _set_latest_frame(annotated, jpeg_quality=70)

        cv2.imshow("Latency Test", annotated)
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


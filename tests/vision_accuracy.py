#!/usr/bin/env python3
"""
Single-file tracker + light + servos + realtime description overlay
+ smoothness logging (angles, velocities, accelerations) to CSV
+ adaptive deadzone logging based on target motion.

Outputs:
  zigzag_smoothness.csv
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import time
import csv
import textwrap
import math

from ultralytics import YOLO
import Jetson.GPIO as GPIO

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

# =========================================================
# LOGGING
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

log_f = open(LOG_PATH, "w", newline="")
log = csv.writer(log_f)
log.writerow([
    "t",
    "pan_deg", "tilt_deg",
    "pan_vel_deg_s", "tilt_vel_deg_s",
    "speed_deg_s",
    "pan_acc_deg_s2", "tilt_acc_deg_s2",
    "human_count", "cx", "cy",
    "deadzone_px",
    "target_motion_px", "target_motion_filt_px",
    "err_x_px", "err_y_px", "err_mag_px",
    "err_x_norm", "err_y_norm", "err_mag_norm"
])

t0 = time.time()
prev_t = None

# Servo derivative state
prev_pan = None
prev_tilt = None
prev_pan_vel = 0.0
prev_tilt_vel = 0.0

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
        t = now - t0

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

        # Defaults for logging
        cx = cy = None
        diff_x = diff_y = 0.0
        err_x = err_y = 0.0
        err_mag_px = 0.0
        err_mag_norm = 0.0
        target_motion_px = 0.0
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

            err_mag_px = math.sqrt(diff_x**2 + diff_y**2)
            err_mag_norm = math.sqrt(err_x**2 + err_y**2)

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

        # ========= DERIVATIVES =========
        pan_vel = tilt_vel = 0.0
        pan_acc = tilt_acc = 0.0
        speed_deg_s = 0.0

        if prev_t is not None:
            dt = t - prev_t
            if dt > 1e-6:
                if prev_pan is None:
                    prev_pan = pan_angle
                    prev_tilt = tilt_angle

                pan_vel = (pan_angle - prev_pan) / dt
                tilt_vel = (tilt_angle - prev_tilt) / dt

                speed_deg_s = math.sqrt(pan_vel**2 + tilt_vel**2)

                pan_acc = (pan_vel - prev_pan_vel) / dt
                tilt_acc = (tilt_vel - prev_tilt_vel) / dt

                prev_pan_vel = pan_vel
                prev_tilt_vel = tilt_vel
                prev_pan = pan_angle
                prev_tilt = tilt_angle

        prev_t = t

        # ========= LOG ROW =========
        log.writerow([
            f"{t:.4f}",
            f"{pan_angle:.3f}", f"{tilt_angle:.3f}",
            f"{pan_vel:.3f}", f"{tilt_vel:.3f}",
            f"{speed_deg_s:.3f}",
            f"{pan_acc:.3f}", f"{tilt_acc:.3f}",
            human_count,
            "" if cx is None else cx,
            "" if cy is None else cy,
            f"{current_deadzone_px:.3f}",
            f"{target_motion_px:.3f}",
            f"{motion_filt_px:.3f}",
            f"{diff_x:.3f}", f"{diff_y:.3f}", f"{err_mag_px:.3f}",
            f"{err_x:.6f}", f"{err_y:.6f}", f"{err_mag_norm:.6f}",
        ])

        # Flush occasionally
        if int(t * 10) % 10 == 0:
            log_f.flush()

        # ========= ANNOTATE + DISPLAY =========
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

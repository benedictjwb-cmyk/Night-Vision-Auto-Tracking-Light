import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 Nano model
print("Loading Model...")
model = YOLO('yolov8n.pt')

# GStreamer pipeline (30 FPS camera, sampling handled in software)
gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)

cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open CSI camera.")
    exit()

print("Starting Human Detection...")

# --- Sampling parameters ---
SAMPLE_PERIOD = 0.25        # seconds
TOTAL_DURATION = 25.0       # seconds
MAX_SAMPLES = int(TOTAL_DURATION / SAMPLE_PERIOD)

sample_count = 0
human_detected_count = 0
last_sample_time = time.time()
start_time = last_sample_time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Run inference continuously (for display)
    results = model(frame, classes=0, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Near-IR Human Detection", annotated_frame)

    # --- Sample every 0.25 s ---
    if current_time - last_sample_time >= SAMPLE_PERIOD:
        last_sample_time = current_time
        sample_count += 1

        # Human present if at least one detection
        human_present = len(results[0].boxes) > 0
        if human_present:
            human_detected_count += 1

        print(f"Sample {sample_count}/100 - Human present: {human_present}")

    # Stop after 25 s (100 samples)
    if sample_count >= MAX_SAMPLES:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Final result ---
percentage = (human_detected_count / sample_count) * 100
print("\n===== Detection Summary =====")
print(f"Total samples: {sample_count}")
print(f"Human detected in: {human_detected_count} samples")
print(f"Detection percentage: {percentage:.1f}%")

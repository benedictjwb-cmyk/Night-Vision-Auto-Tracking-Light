import os
os.chdir(os.path.expanduser("~"))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

import cv2
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge


class VisionYoloCudaStreamNode(Node):
    def __init__(self):
        super().__init__("vision_yolo_cuda_stream_node")

        # Publishers
        self.offset_pub = self.create_publisher(Point, "person_offset", 10)
        self.image_pub = self.create_publisher(Image, "camera/image_raw", 10)

        self.bridge = CvBridge()

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # PyTorch / CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Device: {self.device}")

        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

        # ~30 Hz
        self.timer = self.create_timer(0.03, self.loop)

        self.get_logger().info("Vision YOLO CUDA stream node started")

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        cx_screen = w // 2
        cy_screen = h // 2

        # YOLO inference
        results = self.model(
            frame,
            imgsz=320,
            conf=0.15,
            half=(self.device == "cuda"),
            verbose=False
        )

        # Default offset
        offset_msg = Point()
        offset_msg.x = 0.0
        offset_msg.y = 0.0
        offset_msg.z = 0.0

        boxes = results[0].boxes
        if boxes is not None:
            for i in range(len(boxes)):
                if int(boxes.cls[i]) == 0:  # person
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    dx = cx - cx_screen
                    dy = cy - cy_screen

                    offset_msg.x = float(dx)
                    offset_msg.y = float(dy)

                    # Overlay for livestream
                    cv2.circle(frame, (cx_screen, cy_screen), 6, (255, 0, 0), -1)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.line(frame,
                             (cx_screen, cy_screen),
                             (cx, cy),
                             (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"dx={dx}, dy={dy}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    break

        # Publish offset
        self.offset_pub.publish(offset_msg)

        # Publish image stream
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(img_msg)


def main():
    rclpy.init()
    node = VisionYoloCudaStreamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

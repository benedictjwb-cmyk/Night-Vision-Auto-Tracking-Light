import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

from adafruit_pca9685 import PCA9685
import board
import busio


class ServoNode(Node):
    def __init__(self):
        super().__init__('servo_controller_node')

        self.subscription = self.create_subscription(
            Point,
            'person_offset',
            self.callback,
            10
        )

        self.get_logger().info("Servo controller running")

        # PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50

        self.pan = 135.0
        self.tilt = 135.0

        self.K = 0.05
        self.deadzone = 15

        self.set_servo(0, self.pan)
        self.set_servo(1, self.tilt)

    def set_servo(self, ch, angle):
        angle = max(0, min(270, angle))
        pulse = 500 + (angle / 270.0) * 2000
        duty = int((pulse / 1e6) * 50 * 65536)
        self.pca.channels[ch].duty_cycle = duty

    def callback(self, msg):
        dx = msg.x
        dy = msg.y

        if abs(dx) > self.deadzone:
            self.pan -= self.K * dx
        if abs(dy) > self.deadzone:
            self.tilt += self.K * dy

        self.set_servo(0, self.pan)
        self.set_servo(1, self.tilt)

        self.get_logger().info(
            f"dx={dx:.1f}, dy={dy:.1f} | pan={self.pan:.1f}, tilt={self.tilt:.1f}"
        )


def main():
    rclpy.init()
    node = ServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

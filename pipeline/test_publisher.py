import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
import time

class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.bridge = CvBridge()
        
        # Declare parameters or use default
        self.declare_parameter('video_path', '')
        video_path = self.get_parameter('video_path').get_parameter_value().string_value
        
        if not video_path:
            # Search for any .mp4 files in the directory
            mp4_files = []
            for root, dirs, files in os.walk('.'):
                if 'env' in root or 'venv' in root or '.git' in root:
                    continue
                for file in files:
                    if file.endswith('.mp4'):
                        mp4_files.append(os.path.join(root, file))
            
            if mp4_files:
                video_path = mp4_files[0]
                self.get_logger().info(f"Automatically selected test video: {video_path}")
            else:
                self.get_logger().warn("No .mp4 files found in workspace. Using moving synthetic frames for testing.")
                video_path = None
                
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path) if video_path else None
        
        # Synthetic movement state
        self.square_x = 100
        self.square_y = 150
        self.dx = 5
        self.dy = 3
        
        self.timer = self.create_timer(0.1, self.timer_callback) # 10 FPS (100ms)
        self.get_logger().info("Test Image Publisher initialized! Publishing to /camera/color/image_raw...")

    def timer_callback(self):
        frame = None
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # Loop back to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        
        if frame is None:
            # Generate moving synthetic frames if no video is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw grid
            for i in range(0, 640, 40):
                cv2.line(frame, (i, 0), (i, 480), (30, 30, 30), 1)
            for j in range(0, 480, 40):
                cv2.line(frame, (0, j), (640, j), (30, 30, 30), 1)
            
            # Draw moving "Hand/Body" simulator square to trigger models
            self.square_x += self.dx
            self.square_y += self.dy
            if self.square_x < 50 or self.square_x > 500:
                self.dx = -self.dx
            if self.square_y < 50 or self.square_y > 400:
                self.dy = -self.dy
                
            # Draw simulating elements
            cv2.rectangle(frame, (self.square_x, self.square_y), (self.square_x + 80, self.square_y + 80), (0, 255, 0), -1)
            cv2.circle(frame, (320, 240), 40, (0, 0, 255), 2)
            
            # Status Text
            cv2.putText(frame, "MCN CLOUD TEST ENGINE", (160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Simulating Motion Feed (10 FPS)", (50, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (450, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
        # Convert and publish
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_.publish(img_msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap:
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

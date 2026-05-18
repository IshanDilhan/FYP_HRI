#!/usr/bin/env python3
"""
ROS2 Wrapper Node for the Multimodal Cross-Modal Network (MCN).
Listens to a camera feed, processes frames through the MCN pipeline,
and publishes the resulting behavioral policy as a JSON string.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv_bridge
import json
import cv2
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.video_pipeline import VideoPipeline

class MCNPolicyNode(Node):
    def __init__(self):
        super().__init__('mcn_policy_node')
        
        # ROS2 Parameters
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('policy_topic', '/mcn/behavioral_policy')
        self.declare_parameter('checkpoint_path', 'checkpoints/best_model.pt')
        
        camera_topic = self.get_parameter('camera_topic').value
        policy_topic = self.get_parameter('policy_topic').value
        checkpoint = self.get_parameter('checkpoint_path').value
        
        self.get_logger().info("Initializing MCN Pipeline...")
        # Initialize the actual pipeline (this will load all your custom models)
        self.pipeline = VideoPipeline(mcn_checkpoint=checkpoint)
        
        self.bridge = cv_bridge.CvBridge()
        
        # Subscribe to camera
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10 # QoS profile depth
        )
        self.get_logger().info(f"Subscribed to {camera_topic}")
        
        # Publish policy
        self.publisher = self.create_publisher(String, policy_topic, 10)
        self.get_logger().info(f"Publishing policies to {policy_topic}")
        
        # To avoid running inference on every single 30fps/60fps frame, we can throttle
        self.frame_skip = 2 
        self.frame_count = 0

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return
            
        try:
            # Convert ROS Image to OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Process frame through your custom MCN pipeline
            result = self.pipeline.process_frame(cv_image)
            
            # If a policy was triggered, publish it
            policy = result.get("policy")
            if policy:
                policy_msg = String()
                policy_msg.data = json.dumps(policy)
                self.publisher.publish(policy_msg)
                
                # Beautiful, real-time interactive terminal logs (matching select_and_run.py)
                progress = f"[Frame {self.frame_count}]"
                sys.stdout.write(f"\r{progress} {policy['predicted_intent']:28s} "
                               f"({policy['intent_probability']:.0%}) | "
                               f"E:{result['emotion']['state']:10s} "
                               f"G:{result['gesture']['state']:15s} "
                               f"M:{result['motion']['state']:15s} "
                               f"| {result['fps']:.0f}fps")
                sys.stdout.flush()
                
        except Exception as e:
            sys.stdout.write(f"\n[Error] Failed to process frame: {e}\n")
            sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    node = MCNPolicyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down MCN ROS2 Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

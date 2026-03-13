import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import time
import os
import argparse
import numpy as np
from ultralytics import YOLO

'''
About the whole thing:

Already trained file is "best.pt"

Currently made to run 2 times per second.

I tried to make a "listener" node to help with debugging.
'''



# ==========================================
# NODE 1: Recognizer
# ==========================================
class TrafficSignRecognizer(Node):
    def __init__(self):
        super().__init__('traffic_sign_recognizer')
        
        # Running rate: 2 times per secong
        self.last_process_time = 0.0
        self.process_interval = 0.5 
        
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(String, '/detected_signs', 10)
        self.bridge = CvBridge()

        self.get_logger().info('Loading YOLO Model...')
        
        # To find best.pt in the same folder as this code
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, 'best.pt')
        
        try:
            self.model = YOLO(model_path) 
            self.get_logger().info(f'Model loaded from {model_path}. Node active at 2 Hz!')
        except Exception as e:
            self.get_logger().error(f'FATAL: Could not load model at {model_path}. Error: {e}')
            raise SystemExit

    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_process_time < self.process_interval:
            return 
        
        self.last_process_time = current_time

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        results = self.model(cv_image, verbose=False)
        detected_signs = []
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                if confidence > 0.4:
                    detected_signs.append({"sign": class_name, "confidence": round(confidence, 2)})

        if detected_signs:
            result_msg = String()
            result_msg.data = json.dumps(detected_signs)
            self.publisher_.publish(result_msg)
            self.get_logger().info(f'Published: {result_msg.data}')

# ==========================================
# NODE 2: The Dummy Camera (For Testing)
# ==========================================
class DummyCamera(Node):
    def __init__(self):
        super().__init__('dummy_camera')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.bridge = CvBridge()
        self.frame_count = 0
        self.get_logger().info('Dummy Camera started. Broadcasting fake 30 FPS feed')

    def timer_callback(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        text = f'Dummy Car Feed | Frame: {self.frame_count}'
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.publisher_.publish(msg)
        self.frame_count += 1

# ==========================================
# NODE 3: The Listener (For Debugging)
# ==========================================
class SignListener(Node):
    def __init__(self):
        super().__init__('sign_listener')
        self.subscription = self.create_subscription(
            String, 
            '/detected_signs', 
            self.listener_callback, 
            10)
        self.get_logger().info('Listener active. Waiting for JSON data on /detected_signs')

    def listener_callback(self, msg):
        self.get_logger().info(f'Heard: "{msg.data}"')

# ==========================================
# MAIN
# ==========================================
def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description="Run BFMC Traffic Vision Nodes")
    parser.add_argument('--mode', type=str, choices=['recognize', 'camera', 'listen'], default='recognize', 
                        help="Choose 'recognize' for YOLO, 'camera' for the dummy feed, or 'listen' to read the output.")
    
    # Parse args (ignore ROS specific arguments)
    parsed_args, _ = parser.parse_known_args()
    
    # Launch the requested node
    if parsed_args.mode == 'camera':
        node = DummyCamera()
    elif parsed_args.mode == 'listen':
        node = SignListener()
    else:
        node = TrafficSignRecognizer()
        
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

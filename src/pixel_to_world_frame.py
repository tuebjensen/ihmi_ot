import rospy
import cv2
import cv_bridge
import numpy as np
import sensor_msgs.msg
import matplotlib.pyplot as plt
from  sensor_msgs.msg import Image
from gazebo_msgs.srv import GetModelState
from scipy.spatial.transform import Rotation as R
import time

class PixelToWorldConverter:
    def __init__(self):
        rospy.init_node('pixel_to_world_frame_subscriber', anonymous=True)
        rospy.Subscriber('/camera/depth/image_raw', sensor_msgs.msg.Image, self.callback)
        self.get_kinect_position_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.bridge = cv_bridge.CvBridge()
        self.depth_map = None
        self.width = 640
        self.height = 640
        self.fx = 554.25469
        self.fy = 554.25469
        self.cx = 320.5
        self.cy = 320.5

    def callback(self, msg):
        # Convert ROS Image message to OpenCV image
        self.depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        print(1, self.pixel_to_world(320, 320))

    
    def pixel_to_camera(self, u, v):
        # Perform pixel to world frame conversion
        if self.depth_map is None:
            return None
        z = self.depth_map[v, u]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([z, x, -y], dtype=np.float32)
    
    def pixel_to_world(self, u, v):
        if self.depth_map is None:
            return None
        cam_pose = self.get_kinect_position_srv('kinect', 'world').pose
        cam_position = np.array([cam_pose.position.x, cam_pose.position.y, cam_pose.position.z])
        cam_quat = np.array([cam_pose.orientation.x, cam_pose.orientation.y, cam_pose.orientation.z, cam_pose.orientation.w])
        cam_inv_rotation = R.from_quat(cam_quat).as_matrix()
        # print(cam_position)
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = cam_inv_rotation
        cam_to_world[:3, 3] = cam_position

        p_cam = self.pixel_to_camera(u, v)
        p_world = cam_to_world @ np.concatenate([p_cam, [1]])
        return p_world[:3]

        

if __name__ == '__main__':
    pixel_to_world_frame_subscriber = PixelToWorldConverter()
    rospy.spin()




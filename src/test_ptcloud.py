#!/home/tue/pytorch_env/pytorch_env/bin/python3
import rospy
import cv2
import numpy as np
from  sensor_msgs.msg import PointField, PointCloud2, Image
import std_msgs.msg
import matplotlib.pyplot as plt
import cv_bridge

rospy.init_node('depth_image_subscriber')
pub = rospy.Publisher('/camera/depth/points_scuffed', PointCloud2, queue_size=1)

width = 640
height = 640

fx = 554.25469
fy = 554.25469
cx = 320.5
cy = 320.5

def depth_image_callback(msg):
    # Convert the depth image to an OpenCV image
    ptcloud_msg = PointCloud2()
    ptcloud_msg.header = std_msgs.msg.Header()
    ptcloud_msg.header.stamp = rospy.Time.now()
    ptcloud_msg.header.frame_id = "camera_link"
    ptcloud_msg.height = height
    ptcloud_msg.width = width
    ptcloud_msg.is_dense = True
    fields =[PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        ]
    ptcloud_msg.fields = fields
    ptcloud_msg.point_step = len(fields) * 4
    
    total_num_of_points = ptcloud_msg.height * ptcloud_msg.width
    ptcloud_msg.row_step = ptcloud_msg.point_step * total_num_of_points

    bridge = cv_bridge.CvBridge()
    depth_map = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    data = np.zeros((height, width, len(fields)), dtype=np.float32)
    for v in range(0, height):
        for u in range(0, width):
    #for v, u in [(54, 184), (330, 184), (330, 458), (54, 458)]:
            depth = depth_map[v, u]
            # Get 3D coordinates of pixel (u, v) using the depth value in Gazebo in camera frame
            x = depth
            y = (u - cx) * x / fx
            z = (v - cy) * x / fy
            print(f'depth: {depth}, v: {v}, u: {u}, x: {x}, y: {y}, z: {z}')
            data[v, u] = np.array([x, y, z], dtype=np.float32)
    ptcloud_msg.data = data.tobytes()
    #pub.publish(ptcloud_msg)

            
rospy.Subscriber('/camera/depth/image_raw', Image, depth_image_callback)

# Spin the ROS node to receive messages
rospy.spin()
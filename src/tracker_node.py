#!/home/tue/pytorch_env/pytorch_env/bin/python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from ultralytics import YOLO
import json
import cv_bridge
import numpy as np
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.srv import GetModelState
import sys
classes = {0: 'tennis ball',
           1: 'box'}

class TrackerNode:
    def __init__(self):
        self.model = YOLO(sys.argv[1])
        self.model.to("cuda")
  
        self.obj_pub = rospy.Publisher('detection_data', String, queue_size=10)
        #self.img_pub = rospy.Publisher('annotated_image', Image, queue_size=10)

        self.depth_img_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_img_callback)
        self.color_img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_img_callback)
        self.depth_img = None
        self.depth_img_updated = False
        self.color_img = None
        self.color_img_updated = False
        self.obj_msg = String()
        self.img_msg = Image()

        self.get_kinect_position_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)

        self.cv_bridge = cv_bridge.CvBridge()

        # Camera intrinsics
        # self.width = 640
        # self.height = 640
        self.fx = 554.25469
        self.fy = 554.25469
        self.cx = 320.5
        self.cy = 320.5

        rospy.loginfo('Tracker node started')

    def depth_img_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        self.depth_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_img_updated = True
    
    def color_img_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        self.color_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.color_img_updated = True

    def pixel_to_camera(self, u, v):
        """ Given a pixel coordinate, return the corresponding camera frame coordinates """
        # Perform pixel to camera frame conversion
        z = self.depth_img[v, u]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([z, -x, -y], dtype=np.float32)

    def pixel_to_world(self, u, v):
        """ Given a pixel coordinate, return the corresponding world frame coordinates """
        # Get camera pose
        cam_pose = self.get_kinect_position_srv('kinect', 'world').pose 
        cam_position = np.array([cam_pose.position.x, cam_pose.position.y, cam_pose.position.z])
        cam_quat = np.array([cam_pose.orientation.x, cam_pose.orientation.y, cam_pose.orientation.z, cam_pose.orientation.w])
        cam_rotation = R.from_quat(cam_quat).as_matrix()
        
        # Make homogenous transformation matrix 
        # Vector in camera frame is rotated to be in line with camera orientation and then used to translate to world coordinate of object 
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = cam_rotation
        cam_to_world[:3, 3] = cam_position

        p_cam = self.pixel_to_camera(u, v)
        p_world = cam_to_world @ np.concatenate([p_cam, [1]])
        return p_world[:3]/p_world[3]

    def bboxes_to_json(self, bboxes):
        msg_data = []
        bboxes_cls = bboxes.cls.int().cpu().tolist()
        bboxes_label = [classes[i] for i in bboxes_cls]
        bboxes_conf = bboxes.conf.cpu().tolist()
        tracking_ids = None
        bboxes_xywh = bboxes.xywh.cpu().tolist()
        if bboxes.id is not None:
            tracking_ids = bboxes.id.int().cpu().tolist()
        for i in range(len(bboxes_cls)):
            box = {}
            box['class'] = bboxes_cls[i]
            box['label'] = bboxes_label[i]
            box['confidence'] = bboxes_conf[i]
            box['tracking_id'] = tracking_ids[i] if tracking_ids is not None else None
            box['boundingBox'] = {'x': bboxes_xywh[i][0], 'y': bboxes_xywh[i][1], 'w': bboxes_xywh[i][2], 'h': bboxes_xywh[i][3]}
            world_coordinates = self.pixel_to_world(int(bboxes_xywh[i][0]), int(bboxes_xywh[i][1]))
            box['worldCoordinates'] = {'x': world_coordinates[0], 'y': world_coordinates[1], 'z': world_coordinates[2]}
            msg_data.append(box)

        return json.dumps(msg_data)

    def run(self):
        while not rospy.is_shutdown():
            if not self.color_img_updated:
                continue

            results = self.model.track(self.color_img, persist=True, verbose=False, classes=[0, 1], conf=0.5)
            bboxes = results[0].boxes

            # annotated_frame = results[0].plot()
            # cv2.imwrite('annotated_frame.jpg', annotated_frame)
            
            self.obj_msg.data = self.bboxes_to_json(bboxes)
            # self.img_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.obj_pub.publish(self.obj_msg)
            # self.img_pub.publish(self.img_msg)
            self.color_img_updated = False
            self.depth_img_updated = False 
            rospy.loginfo('Published detection data and annotated image')


if __name__ == '__main__':
    rospy.init_node('tracker_node', anonymous=True)
    tracker_node = TrackerNode()
    tracker_node.run()

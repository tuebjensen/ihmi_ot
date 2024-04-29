#!/home/tue/pytorch_env/pytorch_env/bin/python3
import rospy
import cv2
import object_tracker.msg
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
  
        self.pub = rospy.Publisher('detection_data', String, queue_size=10)
        self.rate = rospy.Rate(20)

        self.depth_map_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_map_callback)
        self.color_img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_img_callback)
        self.depth_map = None
        self.depth_map_updated = False
        self.color_img = None
        self.color_img_updated = False
        self.msg = String()

        self.get_kinect_position_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)

        self.cv_bridge = cv_bridge.CvBridge()

        self.width = 640
        self.height = 640
        self.fx = 554.25469
        self.fy = 554.25469
        self.cx = 320.5
        self.cy = 320.5
        # self.msg.bboxes_cls = object_tracker.msg.UInt16List()
        # self.msg.bboxes_conf = object_tracker.msg.Float32List()
        # self.msg.bboxes_xywh = object_tracker.msg.Float32List()
        # self.msg.tracking_ids = object_tracker.msg.UInt16List()
        rospy.loginfo('Tracker node started')

    def depth_map_callback(self, msg):
        self.depth_map = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_map_updated = True
    
    def color_img_callback(self, msg):
        self.color_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.color_img_updated = True

    def pixel_to_camera(self, u, v):

        # Perform pixel to world frame conversion
        z = self.depth_map[v, u]
        #if label == 'tennis ball':
        #    z = z + 0.035 / 2
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([z, x, -y], dtype=np.float32)

    def pixel_to_world(self, u, v):
        cam_pose = self.get_kinect_position_srv('kinect', 'world').pose 
        cam_position = np.array([cam_pose.position.x, cam_pose.position.y, cam_pose.position.z])
        cam_quat = np.array([cam_pose.orientation.x, cam_pose.orientation.y, cam_pose.orientation.z, cam_pose.orientation.w])
        cam_rotation = R.from_quat(cam_quat).inv()#.as_matrix()
        # print(cam_position)
        # cam_to_world = np.eye(4)
        # cam_to_world[:3, :3] = cam_rotation
        # cam_to_world[:3, 3] = cam_position

        p_cam = self.pixel_to_camera(u, v)
        p_cam_rot = cam_rotation.apply(p_cam)

        p_world = p_cam_rot+cam_position#cam_to_world @ np.concatenate([p_cam, [1]])
        return p_world[:3]

    def boxes_to_json(self, bboxes_cls, bboxes_label, bboxes_conf, tracking_ids, bboxes_xywh):
        msg_data = []
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

        print(msg_data)
        return json.dumps(msg_data)

    def run(self):
        while not rospy.is_shutdown():
            if self.color_img_updated:
                results = self.model.track(self.color_img, persist=True, verbose=False, classes=[0, 1], conf=0.5)
                boxes = results[0].boxes
                bboxes_cls = boxes.cls.int().cpu().tolist()
                bboxes_label = [classes[i] for i in bboxes_cls]
                bboxes_conf = boxes.conf.cpu().tolist()
                tracking_ids = None
                bboxes_xywh = boxes.xywh.cpu().tolist()
                if boxes.id is not None:
                    tracking_ids = boxes.id.int().cpu().tolist()


                self.msg.data = self.boxes_to_json(bboxes_cls, bboxes_label, bboxes_conf, tracking_ids, bboxes_xywh)
                annotated_frame = results[0].plot()
                cv2.imwrite('annotated_frame.jpg', annotated_frame)
                
                self.pub.publish(self.msg)
                self.color_img_updated = False
                self.depth_map_updated = False 
                self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('tracker_node', anonymous=True)
    tracker_node = TrackerNode()
    tracker_node.run()

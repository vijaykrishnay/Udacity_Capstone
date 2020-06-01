#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import os
import time

from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.lights = []
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.pred_count = 0

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        
        self.out_images_debug_path = '/home/workspace/out_imgs'
        if os.path.exists(self.out_images_debug_path):
            os.removedirs(self.out_images_debug_path)
            os.makedirs(self.out_images_debug_path)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
#         rospy.loginfo("Processing traffic lights")
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x ,y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        
        if all(i is not None for i in [self.pose and self.waypoint_tree]):
            closest_idx = self.waypoint_tree.query([x, y],1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN
        
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
#         cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv_image = cv2.resize(cv_image, (224, 224))
        cv_image = cv_image/255.

        #Get classification
        t0 = time.time()
        light_id = self.light_classifier.get_classification(cv_image)
        print("Pred time: ", time.time() - t0)
        
        def get_light_text(id):
            if id == TrafficLight.GREEN:
                return "GREEN"
            elif id == TrafficLight.RED:
                return "RED"
            elif id == TrafficLight.UNKNOWN:
                return "UNKNOWN"
            elif id == TrafficLight.YELLOW:
                return "YELLOW"

        print('Predicted State: ', get_light_text(light_id))
        print('Actual State: ', get_light_text(light.state))

#         # Saving the image
#         self.pred_count += 1
#         file_name = os.path.join(self.out_images_debug_path, 'img_{}_{}.png'.format(self.pred_count, light_id))
#         cv2.imwrite(file_name, cv_image)
        
        return light_id

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_light = None 
        line_wp_idx = None 

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            diff = 50 # TODO: Replace hardcoding with param
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    print("Diff, d:", diff, d)
                    diff = d 
                    closest_light = light 
                    line_wp_idx = temp_wp_idx

        if closest_light:
#             rospy.loginfo("Closest light found")
            state = self.get_light_state(closest_light)
            # print('State: ',state)
            return line_wp_idx, state
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

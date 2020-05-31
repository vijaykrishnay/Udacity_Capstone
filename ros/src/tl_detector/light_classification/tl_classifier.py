from styx_msgs.msg import TrafficLight
import keras
import os
import numpy as np
from keras.utils.generic_utils import CustomObjectScope
import tensorflow as tf
graph = tf.get_default_graph()


class TLClassifier(object):
    def __init__(self):
        CWD_PATH = os.getcwd()
        PATH_TO_h5 = os.path.join(CWD_PATH, 'light_classification', 'model-best.h5')
        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            self.model = keras.models.load_model(PATH_TO_h5)
#         self.model = keras.models.load_model(PATH_TO_h5)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        with graph.as_default():
		#TODO implement light color prediction
		image_expanded = np.expand_dims(image, axis=0)
		#print("Image shape: ",image_expanded.shape)
		prediction = self.model.predict(image_expanded)
		print('Prediction: ',prediction)
                Color = np.argmax(prediction, axis = 1)
                #print('Color Index: ',Color)
		traffic_light_clor = TrafficLight.UNKNOWN
		if Color[0] == 0:
		    traffic_light_clor = TrafficLight.RED
		elif Color[0] == 1:
		    traffic_light_clor = TrafficLight.YELLOW
		elif Color[0] == 2:
		    traffic_light_clor = TrafficLight.GREEN
		elif Color[0] == 3:
		    traffic_light_clor = TrafficLight.UNKNOWN
		# rospy.loginfo("TL Pred: {}".format(prediction))
		# rospy.loginfo("TL Color: {}".format(traffic_light_clor))
                # rospy.loginfo("TL Color Index: {}".format(Color))
        
        return traffic_light_clor

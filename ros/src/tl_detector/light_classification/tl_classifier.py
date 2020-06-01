from styx_msgs.msg import TrafficLight
import keras
import os
import numpy as np
from keras.utils.generic_utils import CustomObjectScope
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        CWD_PATH = os.getcwd()
        PATH_TO_h5 = os.path.join(CWD_PATH, 'light_classification', 'model-best.h5')
        self.graph = tf.get_default_graph()
        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            self.model = keras.models.load_model(PATH_TO_h5)
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # Light color prediction
        image_expanded = np.expand_dims(image, axis=0)
#         print("Image shape: ",image_expanded.shape)
        with self.graph.as_default():
            prediction = self.model.predict(image_expanded)
        
        # Convert pred prob to class id
        predicted_class = np.argmax(prediction, axis=1)
#         print('Prediction: ', prediction)
        # print('Class: ', predicted_class)
        
        # Map class id to traffic light id
        traffic_light_clor = TrafficLight.UNKNOWN
        if predicted_class == 0:
            traffic_light_clor = TrafficLight.GREEN
        elif predicted_class == 1:
            traffic_light_clor = TrafficLight.RED
        elif predicted_class == 2:
            traffic_light_clor = TrafficLight.UNKNOWN
        elif predicted_class == 3:
            traffic_light_clor = TrafficLight.YELLOW
        return traffic_light_clor

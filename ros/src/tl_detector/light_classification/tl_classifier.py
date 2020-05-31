from styx_msgs.msg import TrafficLight
import keras
import os
import numpy as np
from keras.utils.generic_utils import CustomObjectScope


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
        # Light color prediction
        image_expanded = np.expand_dims(image, axis=0)
        print("Image shape: ",image_expanded.shape)
        prediction = self.model.predict(image_expanded)
        print('Prediction: ', prediction)
        traffic_light_clor = TrafficLight.UNKNOWN
        if prediction == 0:
            traffic_light_clor = TrafficLight.GREEN
        elif prediction == 1:
            traffic_light_clor = TrafficLight.RED
        elif prediction == 2:
            traffic_light_clor = TrafficLight.YELLOW
        elif prediction == 3:
            traffic_light_clor = TrafficLight.UNKNOWN
        return traffic_light_clor

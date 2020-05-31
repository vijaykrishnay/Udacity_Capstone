from styx_msgs.msg import TrafficLight
import keras
import numpy as np

class TLClassifier(object):
    def __init__(self):
        CWD_PATH = os.getcwd()
        PATH_TO_h5 = os.path.join(CWD_PATH, 'light_classification', 'model-best.h5')
        model = keras.load(PATH_TO_h5)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_expanded = np.expand_dims(image, axis=0)
        print("Image shape: ",image_expanded.shape)
        prediction = model.predict(image_expanded)
        print('Prediction: ',prediction)

        return TrafficLight.UNKNOWN

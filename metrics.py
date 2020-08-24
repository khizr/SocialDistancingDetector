"""
Implementations of metrics for the social distancing detector.
"""
import abc
import math
import cv2


def eucledian_dist(p1, p2):
    return math.sqrt(abs(p1[0] - p2[0])**2 + abs(p1[1] - p2[1])**2)


class Metric(abc.ABC):
    """
    A metric for evaluating the social distance detector based on the
    boxes of detected objects as well as other details.

    Some metrics may also be visualizable depending on inputs.
    """
    @abc.abstractmethod
    def log_metric(self, img, boxes, logger, details={}):
        """
        Logs the metric's details for the given frame. Depending on the
        metric, img may also be modified.
        """
        pass


class EucledianDistanceViolationMetric(Metric):
    """
    Evaluates whether there exists a pair of objects which have eucledian
    distances (at the bottom centre of the boxes) are less than a set threshold.
    """
    METRIC_KEY = 'eucledian_distance_violation'

    def __init__(self, threshold, visualize=False):
        """
        :threshold: violation threshold in pixels.
        """
        self.threshold = threshold
        self.visualize = visualize

    def log_metric(self, img, boxes, homography, logger, details={}):
        violation_exists = False

        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                box1 = boxes[i]
                mid1 = (int(box1[0] + (box1[2] - box1[0])/2), int(box1[3]))
                t_mid1 = homography.transformed_position(mid1)
                box2 = boxes[j]
                mid2 = (int(box2[0] + (box2[2] - box2[0])/2), int(box2[3]))
                t_mid2 = homography.transformed_position(mid2)

                if eucledian_dist(t_mid1, t_mid2) * homography.pixel_ratio < self.threshold:
                    if not self.visualize:
                        logger.log['metrics'][self.METRIC_KEY].append(True)
                        return
                    violation_exists = True
                    cv2.line(img, mid1, mid2, (0, 0, 255), 2)
                    x1, y1, x2, y2 = box1
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    x1, y1, x2, y2 = box2
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


        logger.log['metrics'][self.METRIC_KEY].append(violation_exists)

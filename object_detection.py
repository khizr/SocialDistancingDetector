"""
Implementations of object detectors.
"""
import abc
import numpy as np

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from darknetpy.detector import Detector


class ObjectDetector(abc.ABC):
    @abc.abstractmethod
    def detect_humans(self, img, img_path, logger):
        """ Returns the object detector's prediction for the image."""
        pass

    @abc.abstractmethod
    def get_human_boxes(self, outputs, logger):
        """
        Given the predictor's output for an image, returns a list of
        bounding box tuples in the form (x1, y1, x2, y2) for the humans in an
        image, where x1 <= x2 and y1 <= y2.
        """
        pass


class FasterRCNNObjectDetector(ObjectDetector):
    MODEL_ZOO_CONFIG = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    MODEL_ZOO_WEIGHTS = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"

    def __init__(self, use_cpu=False):
        setup_logger()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.MODEL_ZOO_CONFIG))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.MODEL_ZOO_WEIGHTS)

        if use_cpu:
            # Using CPU instead of CUDA in order to enable running on Macs
            cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(cfg)

    def detect_humans(self, img, img_path, logger):
        return self.predictor(img)

    def get_human_boxes(self, outputs, logger):
        classes = outputs['instances'].pred_classes.cpu().numpy()
        indexes = np.where(classes == 0)[0]
        boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        boxes_list = boxes[indexes].tolist()

        logger.log['detected_humans_per_frame'].append(len(boxes_list))
        logger.log['boxes_per_frame'].append(boxes_list)
        return boxes_list


class YOLOThreeObjectDetector(ObjectDetector):
    DARKNET_CFG_COCO_DATA = "./yolo_darknet_cfg/coco.data"
    DARKNET_CFG_YOLO_CFG = "./yolo_darknet_cfg/yolov3.cfg"
    YOLO_WEIGHTS = "./yolov3.weights"

    def __init__(self):
        self.detector = Detector(
            self.DARKNET_CFG_COCO_DATA,
            self.DARKNET_CFG_YOLO_CFG,
            self.YOLO_WEIGHTS
        )

    def detect_humans(self, img, img_path, logger):
        return self.detector.detect(img_path)

    def get_human_boxes(self, outputs, logger):
        boxes_list = [[b['left'], b['top'], b['right'], b['bottom']] for b in outputs
            if b['class'] == 'person' and b['prob'] >= 0.95]
        logger.log['detected_humans_per_frame'].append(len(boxes_list))
        logger.log['boxes_per_frame'].append(boxes_list)
        return boxes_list

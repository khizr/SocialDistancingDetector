#!/usr/bin/env python3
import argparse
import cv2
import os
import sys
import json
from collections import defaultdict

import object_detection
import metrics as metrics_module
import homography as homography_module
import calibration as calibration_module


class Logger(object):
    """
    Encapsulates all information pertaining to logging throughout the detector,
    as well as all logged information.

    Example logged information includes the number of frames across a video,
    the number of people detected in each image, etc.
    """
    def __init__(self, log_filepath):
        self.log = {
            'frame_count': 0,
            'detected_humans_per_frame': [],
            'boxes_per_frame': [],
            'metrics': defaultdict(list),
            'homography': {}
        }
        self.filepath = log_filepath

    def write_logs(self):
        with open(self.filepath, 'w') as f:
            f.write(json.dumps(self.log, indent=4, sort_keys=True))


def get_frames(video_path, max_frames, logger):
    """ Extract and return a list of image frames from a video."""
    frames = []
    video = cv2.VideoCapture(video_path)
    success = 1
    frame_count = 0
    while success and frame_count < max_frames:
        success, img = video.read()
        frames.append(img)
        frame_count += 1
    logger.log['frame_count'] = len(frames)
    return frames


def get_fps(video_path):
    video = cv2.VideoCapture(video_path)
    return video.get(cv2.CAP_PROP_FPS)


def get_framecount(video_path):
    video = cv2.VideoCapture(video_path)
    return video.get(cv2.CAP_PROP_FRAME_COUNT)


def get_unlabelled_frames_filepath(video_path):
    return 'frames_{}'.format(video_path.replace('.mp4', '').replace('/', ''))


def get_labelled_frames_filepath(video_path):
    return ('labelled_frames_{}'.format(video_path.replace('.mp4', '')
        .replace('/', '')))


def get_labelled_video_filepath(video_path):
    return ('labelled_video.avi'.format(video_path))


def cache_video_to_frames(frames_dir, video_path, max_frames, logger):
    """
    Converts a video to frames and saves the frames to frames_dir if
    frames_dir doesn't already exist.
    """
    if frames_dir not in os.listdir('.') or len(os.listdir(frames_dir)) != max_frames:
        if frames_dir not in os.listdir('.'):
            os.mkdir(frames_dir)
        frames = get_frames(video_path, max_frames, logger)
        for i, img in enumerate(frames):
            frame_path = os.path.join(frames_dir, '{}.jpg'.format(i))
            cv2.imwrite(frame_path, img)


def cache_frames_to_video(labelled_frames_dir, frames_count, fps, video_filepath):
    """
    Converts frames to a video and writes the video to video_filepath if it
    doesn't already exist.
    """
    if video_filepath not in os.listdir('.'):
        frame = cv2.imread(os.path.join(labelled_frames_dir, '0.jpg'))
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video = cv2.VideoWriter(video_filepath, fourcc, fps, (width, height))
        video.write(frame)

        for i in range(1, frames_count):
            frame = cv2.imread(os.path.join(labelled_frames_dir, '{}.jpg'.format(i)))
            video.write(frame)
        video.release()


def process_image(img, img_path, object_detector, metrics, homography, logger):
    """ Processes one image/frame in the social distancing detector."""
    outputs = obj_detector.detect_humans(img, img_path, logger)
    boxes = obj_detector.get_human_boxes(outputs, logger)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    for metric in metrics:
        metric.log_metric(img, boxes, homography, logger, details={})

    return img


def run_detector(video_path, object_detector, metrics, max_frames, homography, logger):
    frames_dir = get_unlabelled_frames_filepath(video_path)
    max_frames = min(max_frames, get_framecount(video_path))
    cache_video_to_frames(frames_dir, video_path, max_frames, logger)
    fps = get_fps(video_path)

    labelled_frames_dir = get_labelled_frames_filepath(video_path)
    if labelled_frames_dir not in os.listdir('.'):
        os.mkdir(labelled_frames_dir)

    frames_count = min(len(os.listdir(frames_dir)),  max_frames)
    for i in range(frames_count):
        frame_path = os.path.join(frames_dir, '{}.jpg'.format(i))
        frame = cv2.imread(frame_path)

        process_image(frame, frame_path, object_detector, metrics, homography, logger)
        print('Finished processing frame {}/{}.'.format(i+1, frames_count))

        labelled_frame_path = os.path.join(labelled_frames_dir, '{}.jpg'.format(i))
        cv2.imwrite(labelled_frame_path, frame)

    labelled_video_path = get_labelled_video_filepath(video_path)
    cache_frames_to_video(labelled_frames_dir, frames_count, fps, labelled_video_path)

    logger.write_logs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform social distancing detection on a video.')
    parser.add_argument('video_path', help="Path to the video file")
    parser.add_argument('obj_det_method', help="The object detection method ('YOLO' or 'RCNN')")
    parser.add_argument(
        'max_frames', help="The maximum number of frames to process. Set to -1 to process all frames.")
    parser.add_argument('log_path', help="The path to save the JSON file containing the logs.")
    parser.add_argument('use_cpu',
        help="For Faster RCNN object detection, runs on CPU if set to true. Must be set to true to run on Mac.")
    args = parser.parse_args()

    if args.obj_det_method == 'RCNN':
        if args.use_cpu.lower() == 'true':
            obj_detector = object_detection.FasterRCNNObjectDetector(use_cpu=True)
        else:
            obj_detector = object_detection.FasterRCNNObjectDetector()
    elif args.obj_det_method == 'YOLO':
        obj_detector = object_detection.YOLOThreeObjectDetector()
    else:
        print('Invalid object detection approach selected.')
        exit(1)

    args.max_frames = int(args.max_frames)
    if args.max_frames == -1:
        args.max_frames = sys.maxsize

    logger = Logger(args.log_path)

    vid = cv2.VideoCapture(args.video_path)
    success, first_frame = vid.read()
    calibration = calibration_module.Calibration(first_frame)
    calibration.calibrate()
    corners, width_in_metres, height_in_metres = calibration.getData()

    homography = homography_module.Homography(corners, width_in_metres, height_in_metres)
    homography.log_data(logger)

    # These are the extra metrics which will be logged
    metrics = ([
        metrics_module.EucledianDistanceViolationMetric(threshold=2, visualize=True)
    ])

    run_detector(
        args.video_path,
        obj_detector,
        metrics,
        args.max_frames,
        homography,
        logger
    )

#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import math

import homography as hg

LOGS_DIR = './experiment_logs'


def get_ax(axs, index):
    choices = {
        0: axs[0, 0],
        1: axs[0, 1],
        2: axs[1, 0],
        3: axs[1, 1]
    }
    return choices[index]


def load_logs():
    logs = {}
    for fname in os.listdir(LOGS_DIR):
        with open(os.path.join(LOGS_DIR, fname)) as f:
            logs[fname.replace('.json', '')] = json.load(f)

    return [
        (logs['oxford_rcnn'], logs['oxford_yolo']),
        (logs['sports_rcnn'], logs['sports_yolo']),
        (logs['virat_rcnn'], logs['virat_yolo']),
        (logs['peshawar_rcnn'], logs['peshawar_yolo'])
    ]


def get_homography(logs):
    return ([hg.Homography(
            log_tup[0]['homography']['corners'],
            log_tup[0]['homography']['width'],
            log_tup[0]['homography']['height'])
        for log_tup in logs])


def count_violating_pairs(boxes_ls, homography):
    def eucledian_dist(p1, p2):
        return math.sqrt(abs(p1[0] - p2[0])**2 + abs(p1[1] - p2[1])**2)

    result = []
    for boxes in boxes_ls:
        # Modified code from Metrics which counts violations, produces the same results
        count = 0
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                box1 = boxes[i]
                mid1 = (int(box1[0] + (box1[2] - box1[0])/2), int(box1[3]))
                t_mid1 = homography.transformed_position(mid1)
                box2 = boxes[j]
                mid2 = (int(box2[0] + (box2[2] - box2[0])/2), int(box2[3]))
                t_mid2 = homography.transformed_position(mid2)

                if eucledian_dist(t_mid1, t_mid2) * homography.pixel_ratio < 2:
                    count += 1
        result.append(count)
    return result


def plot_violation_count(logs):
    axs = get_plot("Frame", "Violating Pairs", "Detected Violating Pairs - Faster RCNN (blue) vs YOLOv3 (orange)")
    homography = get_homography(logs)

    for i in range(len(logs)):
        rcnn_boxes = logs[i][0]["boxes_per_frame"]
        yolo_boxes = logs[i][1]["boxes_per_frame"]

        x1 = range(len(rcnn_boxes))
        y1 = count_violating_pairs(rcnn_boxes, homography[i])
        get_ax(axs, i).plot(x1, y1, label='Faster RCNN')

        x2 = range(len(yolo_boxes))
        y2 = count_violating_pairs(yolo_boxes, homography[i])
        get_ax(axs, i).plot(x2, y2, label='YOLOv3')

    plt.show()


def print_violation_ratios(logs):
    homography = get_homography(logs)

    total_rcnn_count = 0
    total_yolo_count = 0
    for i in range(len(logs)):
        rcnn_boxes = logs[i][0]["boxes_per_frame"]
        yolo_boxes = logs[i][1]["boxes_per_frame"]

        rcnn_count = sum(count_violating_pairs(rcnn_boxes, homography[i]))
        yolo_count = sum(count_violating_pairs(yolo_boxes, homography[i]))
        total_rcnn_count += rcnn_count
        total_yolo_count += yolo_count
        print(float(rcnn_count)/yolo_count)
    print(float(total_rcnn_count)/total_yolo_count)


def get_plot(xlabel, ylabel, supertitle):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(supertitle, fontsize=20)
    for ax in axs.flat:
        ax.set(xlabel=xlabel, ylabel=ylabel)

    axs[0, 0].set_title('Oxford Town Centre Dataset')
    axs[0, 1].set_title('Sports Hall Dataset')
    axs[1, 0].set_title('VIRAT Dataset')
    axs[1, 1].set_title('Peshawar Dataset')
    return axs


def plot_detected_people(logs):
    axs = get_plot("Frame", "Detected People", "Detected People - Faster RCNN (blue) vs YOLOv3 (orange)")
    for i in range(len(logs)):
        rcnn_boxes = logs[i][0]["boxes_per_frame"]
        yolo_boxes = logs[i][1]["boxes_per_frame"]

        x1 = range(len(rcnn_boxes))
        y1 = [len(ls) for ls in rcnn_boxes]
        get_ax(axs, i).plot(x1, y1, label='Faster RCNN')

        x2 = range(len(yolo_boxes))
        y2 = [len(ls) for ls in yolo_boxes]
        get_ax(axs, i).plot(x2, y2, label='YOLOv3')

    plt.show()


def print_violations(logs):
    def count_log(log):
        return (log[0]["metrics"]["eucledian_distance_violation"].count(True),
            log[1]["metrics"]["eucledian_distance_violation"].count(True),
            len(log[1]["metrics"]["eucledian_distance_violation"]))
    for log in logs:
        print(count_log(log))


if __name__ == '__main__':
    logs = load_logs()
    # print_violation_ratios(logs)
    plot_violation_count(logs)
    # plot_detected_people(logs)
    # print_violations(logs)

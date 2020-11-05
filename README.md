# Introduction

Built a framework which takes as input a video and performs social distancing detection. This framework detects humans within the video, applies homography techniques to adjust for depth and distance, detects whether humans are at least a safe distance of each other (2 meters), and outputs a labelled video in which social distancing violations are visually marked. This framework implements and allows two leading object detection approaches to be used interchangeably (Faster R-CNNs and YOLOv3). Information is logged as it runs, thus allowing for comparisons between the two approaches. 

We use this framework to detect social distancing in 4 different video datasets with both Faster R-CNNs and YOLOv3, and perform analysis on the data. We derive several insights on social distance detection from this data and conclude with a comparison on the performance of Faster R-CNNs versus YOLOv3 for social distancing detection, and we finally present that a social distancing detector using Faster R-CNNs correctly detects social distancing violations 2.52 times more frequently than YOLOv3.

**Labelled Output**
\
![](output.gif)


# Installation

**Google drive containing RCNN inputs and results:***
https://drive.google.com/drive/folders/1xJaAs5CqOp1v7XIeuWg6aY2MvpPEFRP0?usp=sharing

**Google drive containing YOLOv3 inputs, weights file, and results:***
https://drive.google.com/drive/folders/1AdVufg4cprv0mZhkhSBLyFnoJU6aM5rH?usp=sharing

**Install dependencies: (use cu101 because colab has CUDA 10.1)**
!pip install cython pyyaml==5.1

**Install detectron2**
!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

Note: if the above doesn't work, try pip3 install torch torchvision
Note: install a CUDA version of torch, see https://pytorch.org

**Install darknetpy (Object detection with YOLO)**
https://pypi.org/project/darknetpy/

Then download weights file for "yolov3-608" from
https://pjreddie.com/darknet/yolo/
or download the file from our YOLOv3 Google Drive (labelled yolov3.weights) and
place it in the same directory as detector.py. (The file is 250MB and is too large
to directly submit)

**Get info on script:** ./detector.py -h

**Example script run command (RCNN using CPU):** ./detector.py sample.mp4 RCNN -1 logs.json true

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button


class Calibration:
    def __init__(self, img):
        self.img = img
        self.corners = []
        self.width = 0
        self.height = 0

    def anounce(self, s):
        plt.title(s, fontsize=13)
        plt.draw()

    def calibrate(self):

        plt.clf()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.imshow(self.img)

        self.anounce('You will define a rectangle in the image \n by selecting its corners, click to begin')
        plt.waitforbuttonpress()

        while True:
            pts = []
            while len(pts) < 3:
                self.anounce('Select 4 corners of the rectangle in the following order:\n top left, top right, bottom left, bottom right')
                pts = np.asarray(plt.ginput(4, timeout=-1))
                if len(pts) < 4:
                    self.anounce('Too few points, starting over')

            self.anounce('Happy? Key click for yes, mouse click for no')

            fill_pts = np.copy(pts)
            fill_pts[[2, 3]] = fill_pts[[3, 2]]
            print(pts)
            print(fill_pts)
            ph = plt.fill(fill_pts[:, 0], fill_pts[:, 1], 'r', lw=2)

            if plt.waitforbuttonpress():
                break

            for p in ph:
                p.remove()

        pts = pts.tolist()
        for pt in pts:
            pt[0] = round(pt[0])
            pt[1] = round(pt[1])
        self.corners = pts

        def submitWidth(text):
            self.width = eval(text)

        def submitHeight(text):
            self.height = eval(text)

        def submit(event):
            print("Corners: {}".format(self.corners))
            print("Width is {}m".format(self.width))
            print("Height is {}m".format(self.height))
            print("Calibration is complete")
            plt.close('all')

        self.anounce("Enter the width and height (in metres) in the fields below \n Width: top left corner to top right corner \n Height: top left corner to bottom left corner")
        width_box = TextBox(plt.axes([0.2, 0.02, 0.2, 0.05]), 'Width:', initial='')
        width_box.on_text_change(submitWidth)
        height_box = TextBox(plt.axes([0.6, 0.02, 0.2, 0.05]), 'Height:', initial='')
        height_box.on_text_change(submitHeight)

        submit_button = Button(plt.axes([0.85, 0.02, 0.1, 0.05]), 'Submit')
        submit_button.on_clicked(submit)

        plt.show()

    def getData(self):
        return self.corners, self.width, self.height
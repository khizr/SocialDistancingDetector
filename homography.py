import numpy as np
import math

class Homography:

    def __init__(self, corners, width, height):
        self.corners = corners
        self.width = width
        self.height = height
        self.h = self.find_homography()
        self.pixel_ratio = self.find_pixel_ratio()

    def dist(self, x1, y1, x2, y2):
        return int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    def find_pixel_ratio(self):
        c0 = self.transformed_position(self.corners[0])
        c1 = self.transformed_position(self.corners[1])
        c2 = self.transformed_position(self.corners[2])

        t_width_pixels = self.dist(c0[0], c0[1], c1[0], c1[1])
        t_height_pixels = self.dist(c0[0], c0[1], c2[0], c2[1])

        ratio_1 = self.width / t_width_pixels
        ratio_2 = self.height / t_height_pixels
        return (ratio_1 + ratio_2) / 2

    # Returns homography matrix given the 4 corner coordinates of a rectangle in the image
    # corners is a list of lists... [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # corners MUST be in order (top left, top right, bottom left, bottom right)
    def find_homography(self):
        x1 = self.corners[0][0]
        y1 = self.corners[0][1]
        x2 = self.corners[1][0]
        y2 = self.corners[1][1]
        x3 = self.corners[2][0]
        y3 = self.corners[2][1]
        x4 = self.corners[3][0]
        y4 = self.corners[3][1]

        xdist_1 = self.dist(x1, y1, x2, y2)
        xdist_2 = self.dist(x3, y3, x4, y4)
        ydist_1 = self.dist(x1, y1, x3, y3)
        ydist_2 = self.dist(x2, y2, x4, y4)

        w = max(xdist_1, xdist_2)
        h = max(ydist_1, ydist_2)

        if (w > h):
            h = int(w * (self.height / self.width))
        else:
            w = int(h * (self.width / self.height))

        A = [[x1, y1, 1, 0, 0, 0, -x1, -y1],
             [0, 0, 0, x1, y1, 1, -x1, -y1],
             [x2, y2, 1, 0, 0, 0, -w * x2, -w * y2],
             [0, 0, 0, x2, y2, 1, -x2, -y2],
             [x3, y3, 1, 0, 0, 0, -x3, -y3],
             [0, 0, 0, x3, y3, 1, -h * x3, -h * y3],
             [x4, y4, 1, 0, 0, 0, -w * x4, -w * y4],
             [0, 0, 0, x4, y4, 1, -h * x4, -h * y4]]
        A = np.array(A)

        # b contains the locations of the transformed corner points
        b = np.array([[1],
                      [1],
                      [w],
                      [1],
                      [1],
                      [h],
                      [w],
                      [h]])

        tmp = np.dot(np.linalg.inv(A), b)
        h = np.array([[tmp[0][0], tmp[1][0], tmp[2][0]],
                      [tmp[3][0], tmp[4][0], tmp[5][0]],
                      [tmp[6][0], tmp[7][0], 1]
                      ])
        return h

    def transformed_position(self, position):
        position_array = np.array([[position[0]],
                                   [position[1]],
                                   [1]])
        transformed = np.matmul(self.h, position_array)
        x = transformed[0, 0]
        y = transformed[1, 0]
        a = transformed[2, 0]
        pos = (int(round(x/a)), int(round(y/a)))
        return pos

    def log_data(self, logger):
        logger.log['homography']['corners'] = self.corners
        logger.log['homography']['width'] = self.width
        logger.log['homography']['height'] = self.height
        logger.log['homography']['matrix'] = self.h.tolist()
        logger.log['homography']['metre_per_pixel'] = self.pixel_ratio
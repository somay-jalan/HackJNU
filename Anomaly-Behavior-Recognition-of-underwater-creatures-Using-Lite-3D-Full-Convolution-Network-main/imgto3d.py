import numpy as np
import cv2


class Imgto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def img3d(self, filenames, color=False, skip=True):
        framearray = []
        for fileName in filenames:
            img = cv2.imread(fileName)
            frame = cv2.resize(img, (self.height, self.width))
            if color:
                framearray.append(frame)
            else:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return np.array(framearray)

    def get_UCF_classname(self, filename):
        x = filename[filename.find('_') + 1:filename.find('_', 2)]
        print(x)
        return x

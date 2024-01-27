import numpy as np
from tqdm import tqdm
import os
import cv2
from tensorflow.keras.utils import to_categorical


def loaddata(img_dir, img3d, nclass, result_dir, color=False, skip=True):
    files = os.listdir(img_dir)
    X = []
    labels = []
    labellist = []
    pbar = tqdm(total=len(files))
    for filename in files:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(img_dir, filename)
        for sample_files in os.listdir(name):
            img_file_path = os.path.join(name, sample_files)
            img_files = [f"{img_file_path}/{x}" for x in os.listdir(img_file_path)]
            label = filename

            if label not in labellist:

                if len(labellist) >= nclass:
                    continue

                labellist.append(label)

            labels.append(label)

            X.append(img3d.img3d(img_files, color=color, skip=skip))

    pbar.close()

    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels


def getAugData(X1):
    X2_temp = []
    for x1 in X1:
        xx1 = x1.copy()
        for d in range(10):
            temp = xx1[:, :, d, :]
            xx1[:, :, d, :] = cv2.flip(temp, 1)
        X2_temp.append(xx1)
    X2 = np.array(X2_temp)
    return X2


def getMixAugData(X1, Y1):
    mix_X_temp = []
    for x1 in X1:
        mix_X_temp.append(x1)
    for x2 in getAugData(X1):
        mix_X_temp.append(x2)
    mix_X = np.array(mix_X_temp)
    mix_Y_temp = []
    for _ in range(2):
        for y1 in Y1:
            mix_Y_temp.append(y1)
    mix_Y = np.array(mix_Y_temp)
    return mix_X, mix_Y


def dataPreprocess(x, y, img_rows, img_cols, frames, channel, nb_classes):
    X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    Y = to_categorical(y, nb_classes)
    X = X.astype('float32')
    return X, Y


def loadCategory(output="./asset"):
    file = open(f"{output}/classes.txt")
    labels = file.read()
    labels = labels.split("\n")
    labels.pop(-1)
    file.close()
    return labels

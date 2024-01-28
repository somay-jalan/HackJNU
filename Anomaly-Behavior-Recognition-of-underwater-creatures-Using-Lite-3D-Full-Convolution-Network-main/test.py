import time
import os
import matplotlib
matplotlib.use('AGG')
import numpy as np
from sklearn.model_selection import train_test_split
import imgto3d
from  tensorflow.keras.callbacks import ModelCheckpoint
from  tensorflow.keras.models import load_model
##
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import cv2
import mediapipe as mp
import pyautogui
from utils.focal_loss import *
from utils import dataprocess as dp
from utils import save_report_plot as srp
from utils import models
from multiprocessing import Process

def makeFrames():
    vidcap = cv2.VideoCapture('fishie.mp4')
    frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = vidcap.get(cv2.CAP_PROP_FPS) 
    seconds =(frames / fps)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite("testImage/test/cobia_anomaly/"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = seconds/10 
    count=1
    success = getFrame(sec)
    while success:
        if count>=10:
            count-=10
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

def checkFrames():
    size = 64
    img_rows, img_cols, frames = size, size, 10
    img3d = imgto3d.Imgto3D(img_rows, img_cols, frames)
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    loaded_model=load_model('save_model_ar.hd5')
    x,y=dp.loaddata("testImage",img3d,7,"result/",False,True)
    X, Y = dp.dataPreprocess(x, y, img_rows, img_cols, frames, 1, 7)
    predictions=loaded_model.predict(X)
    # print(predictions)
    classes = np.argmax(predictions, axis=1)
    if classes==[1]:
        print("Cobia_anomaly")
# print(loaded_model.evaluate(X,Y,verbose=0))

makeFrames()
checkFrames()

# processCheckFrames.start()




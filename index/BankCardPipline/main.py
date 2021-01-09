import os
import cv2
import logging
import collections
import time
import datetime
import numpy as np
import uuid
import json
import functools
from .EAST.eval import get_detector
from .CardRecognition.run_model import get_res_from_img
from .CardRecognition.enhance import equal_hist_with_opt
from PIL import Image
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))

detector_path = os.path.join(cur_dir,'EAST/models/model_202.pkl')
recognitionor_path = os.path.join(cur_dir,'CardRecognition/saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth')
detector = get_detector(detector_path)
recognitionor = get_res_from_img(recognitionor_path)

def pipline(im):
    global detector,recognitionor
    
    imgOut,labeled_img = detector(im)
    #cv2.imshow('imgout1',imgOut)
    imgOut = imgOut[:, :, ::-1]
    #cv2.imshow('imgout2',imgOut)
    #cv2.waitKey(0)


    if imgOut is not None:
        image = Image.fromarray(equal_hist_with_opt(img = imgOut))
        pred = recognitionor(image)
        return pred,labeled_img
    return -1,None

if __name__ == '__main__':
    root = 'test_images/djl2.jpg'
    im = cv2.imread(root)
    pred = pipline(im)
    print(root)
    print(pred)
    # imglist = os.listdir(root)
    # for img in imglist :

    #     im_path = root + '/' + img 
    #     im = cv2.imread(im_path)
    #     pred = pipline(im)
    #     print(img)
    #     print(pred)
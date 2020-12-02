import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

import scipy.io as sio
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import matplotlib.pyplot as plt
import matplotlib
from yolo_model import YoloModel
from rcnn_model import FasterRCNN


class ModelStack:

    def __init__(self):
        self.yolo = YoloModel("yolov3.weights", "yolov3.cfg", "coco.names")
        self.rcnn = FasterRCNN("updated_rcnn.zip")

    def get_person_count(self,img,plot=True):
        yolo_prediction = self.yolo.predict(img,plot=plot)
        return yolo_prediction["label_count"]

    def get_mask_count(self, img, plot=True, plot_save_to=None):
        label_frequencies = self.rcnn.predict(img,plot=plot,plot_save_to=plot_save_to)
        return label_frequencies["label_frequencies"]

    def get_mask_compliance(self, img, write_to_path=None,img_path=""):
        label_frequencies = self.get_mask_count(img,plot=False)
        print(label_frequencies)
        yolo_person_count = self.get_person_count(img,plot=False)
        mask_wearing_count = label_frequencies["with_mask"]
        rcnn_person_count = 0
        for label_dict in label_frequencies.items():
            rcnn_person_count += label_dict[1]
        print("RCNN Person Count:",rcnn_person_count)
        print("With Mask Count:", mask_wearing_count)
        print("Incorrect Mask Count:", label_frequencies["mask_weared_incorrect"])
        print("Without Mask Count:", label_frequencies["without_mask"])
        print("YOLO Person Count:", yolo_person_count)
        min_person_count = min(rcnn_person_count, yolo_person_count)
        max_person_count = max(rcnn_person_count, yolo_person_count)
        min_mask_compliance = mask_wearing_count / max_person_count
        max_mask_compliance = mask_wearing_count / min_person_count
        print("Minimum Compliance:", min_mask_compliance)
        print("Maximum Compliance:", max_mask_compliance)
        if(write_to_path):
            with open(write_to_path,'a') as f:
                f.write(img_path + "\n")
                f.write("RCNN Person Count: "+str(rcnn_person_count)+"\n")
                f.write("With Mask Count: "+str(mask_wearing_count)+"\n")
                f.write("Incorrect Mask Count: " +  str(label_frequencies["mask_weared_incorrect"])+"\n")
                f.write("Without Mask Count: " +  str(label_frequencies["without_mask"])+"\n")
                f.write("YOLO Person Count: "+str(yolo_person_count)+"\n")
                f.write("Minimum Compliance: "+str(min_mask_compliance)+"\n")
                f.write("Maximum Compliance: "+str(max_mask_compliance)+"\n\n")

if __name__ == "__main__":

    stack = ModelStack()
    base_dir = "../crowd_images/"
    for path in os.listdir(base_dir):
        if(path != ".DS_Store"):
            print(path)
            img = cv2.cvtColor(cv2.imread(os.path.join(base_dir,path)), cv2.COLOR_BGR2RGB)
            stack.get_mask_count(img,plot=True,plot_save_to=os.path.join("../crowd_predictions/",path))
            stack.get_mask_compliance(img, write_to_path="../crowd_predictions/predictions.txt",img_path=path)
            print()
            print()
    # img = cv2.cvtColor(cv2.imread(os.path.join(base_dir,"mask3.jpeg")), cv2.COLOR_BGR2RGB)
    # stack.get_mask_count(img,plot=True)
    # stack.get_mask_compliance(img)
    # print(stack.get_mask_count(img))    
    



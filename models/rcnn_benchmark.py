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
from crowd_analysis import ModelStack
from collections import Counter

if __name__ == "__main__":

    stack = ModelStack()
    base_dir = "../covid_benchmark_data/train/NonMask"
    non_mask_predicted_frequencies = Counter({"with_mask":0, "mask_weared_incorrect":0,"without_mask":0,"background":0})
    len_dir = len(os.listdir(base_dir)) -1
    index = 0
    for path in os.listdir(base_dir):
        if(path != ".DS_Store"):
            img = cv2.cvtColor(cv2.imread(os.path.join(base_dir,path)), cv2.COLOR_BGR2RGB)
            # plt.imshow(img)
            # plt.show()
            index += 1
            print(index, "/", len_dir)
            predictions = stack.get_mask_predictions(img)
            non_mask_predicted_frequencies += Counter(predictions['label_frequencies'])
    print(dict(non_mask_predicted_frequencies.most_common()))
    # {'without_mask': 275, 'with_mask': 45, 'mask_weared_incorrect': 3}
    # {'with_mask': 299, 'without_mask': 6, 'mask_weared_incorrect': 2}
    index = 0
    mask_predicted_frequencies = Counter({"with_mask":0, "mask_weared_incorrect":0,"without_mask":0,"background":0})
    base_dir = "../covid_benchmark_data/train/Mask"
    len_dir = len(os.listdir(base_dir)) -1
    for path in os.listdir(base_dir):
        if(path != ".DS_Store"):
            img = cv2.cvtColor(cv2.imread(os.path.join(base_dir,path)), cv2.COLOR_BGR2RGB)
            index += 1
            print(index, "/", len_dir)
            predictions = stack.get_mask_predictions(img)
            mask_predicted_frequencies += Counter(predictions['label_frequencies'])
    print(dict(mask_predicted_frequencies.most_common()))
            
            


    



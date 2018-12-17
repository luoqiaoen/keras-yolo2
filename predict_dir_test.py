#! /usr/bin/env python
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes, BoundBox
from frontend import YOLO
import json
import pandas as pd
from math import floor, ceil

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    config_path  = "config.json"
    weights_path = "../large_weights/full_yolo_whale.h5"
    directory_in_str = "../large_dataset/whale_files/test"

    df = pd.read_csv("../large_dataset/whale_files/sample_submission.csv",header=None,names = ["Image", "Id", "Xmin", "Ymin", "Xmax", "Ymax"])
    df = df.drop([0])
    manual_check = pd.DataFrame(columns=["Image", "Id", "Xmin", "Ymin", "Xmax", "Ymax"])

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################
    directory = os.fsencode(directory_in_str)
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(directory_in_str, filename))
            image_h, image_w, _ = image.shape
            boxes = yolo.predict(image)
            if len(boxes) == 1:
                df.at[df.Image == filename, 'Xmin'] = max(floor(boxes[0].xmin*image_w),0)
                df.at[df.Image == filename, 'Ymin'] = max(floor(boxes[0].ymin*image_h),0)
                df.at[df.Image == filename, 'Xmax'] = min(ceil(boxes[0].xmax*image_w),image_w)
                df.at[df.Image == filename, 'Ymax'] = min(ceil(boxes[0].ymax*image_h),image_h)
            else:
                s = df[df.Image == filename]
                manual_check = pd.concat([manual_check,s])
        else:
            continue
    df.to_csv('test_box.csv', encoding='utf-8', index=False)
    manual_check.to_csv('manual_check_test.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    main()

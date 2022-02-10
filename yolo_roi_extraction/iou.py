# reference https://github.com/ultralytics/yolov3

import os
import glob
import numpy as np
from statistics import *
import cv2
import shutil

def get_iou(bb1, bb2, labelPath, predPath):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        print('hhhhh')
        print(labelPath)
        print(predPath)
        print('bb1', bb1)
        print('bb2', bb2)
        print('x_right, x_left, y_bottom, y_top', x_right, x_left, y_bottom, y_top)
        return 0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def iou(path1, path2):
    txtList = os.listdir(path1)
    iouList = []
    for txt in txtList:
        labelPath = path1 + txt
        predPath = path2 + txt
        jpgPath = path2 + txt.replace('.txt', '.jpg')

        if os.path.exists(predPath):
            h, w, _ = cv2.imread(jpgPath).shape
            with open(labelPath, 'r') as f:
                # print(labelPath)
                x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                box1 = x
            with open(predPath, 'r') as ff:
                # print(predPath)
                x = np.array([x.split() for x in ff.read().splitlines()], dtype=np.float32)
                box2 = x
            # print('\n')
            if box1.shape[0] == 2 and box2.shape[0] == 2:
                b2_x1_flag, b2_x2_flag = int(box2[0][0]), int(box2[0][2])
                b2_x1_flag_prime, b2_x2_flag_prime = int(box2[1][0]), int(box2[1][2])

                if b2_x1_flag > b2_x1_flag_prime:
                    b2_x1, b2_x2 = int(box2[1][0]), int(box2[1][2])
                    b2_y1, b2_y2 = int(box2[1][1]), int(box2[1][3])

                    b1_x1, b1_x2 = int(w * (box1[0][1])) - int(w * (box1[0][3]) / 2), int(w * (box1[0][1])) + int(
                        w * (box1[0][3]) / 2)
                    b1_y1, b1_y2 = int(h * (box1[0][2])) - int(h * (box1[0][4]) / 2), int(h * (box1[0][2])) + int(
                        h * (box1[0][4]) / 2)

                    bb1 = {'x1': b1_x1, 'x2': b1_x2, 'y1': b1_y1, 'y2': b1_y2}
                    bb2 = {'x1': b2_x1, 'x2': b2_x2, 'y1': b2_y1, 'y2': b2_y2}
                    iou = get_iou(bb1, bb2, labelPath, predPath)
                    iouList.append(iou)
                    ###########################################################
                    b2_x1, b2_x2 = int(box2[0][0]), int(box2[0][2])
                    b2_y1, b2_y2 = int(box2[0][1]), int(box2[0][3])

                    b1_x1, b1_x2 = int(w * (box1[1][1])) - int(w * (box1[1][3]) / 2), int(w * (box1[1][1])) + int(
                        w * (box1[1][3]) / 2)
                    b1_y1, b1_y2 = int(h * (box1[1][2])) - int(h * (box1[1][4]) / 2), int(h * (box1[1][2])) + int(
                        h * (box1[1][4]) / 2)

                    bb1 = {'x1': b1_x1, 'x2': b1_x2, 'y1': b1_y1, 'y2': b1_y2}
                    bb2 = {'x1': b2_x1, 'x2': b2_x2, 'y1': b2_y1, 'y2': b2_y2}
                    iou = get_iou(bb1, bb2, labelPath, predPath)
                    iouList.append(iou)


                else:
                    b1_x1, b1_x2 = int(w * (box1[0][1])) - int(w * (box1[0][3]) / 2), int(w * (box1[0][1])) + int(
                        w * (box1[0][3]) / 2)
                    b1_y1, b1_y2 = int(h * (box1[0][2])) - int(h * (box1[0][4]) / 2), int(h * (box1[0][2])) + int(
                        h * (box1[0][4]) / 2)
                    b2_x1, b2_x2 = int(box2[0][0]), int(box2[0][2])
                    b2_y1, b2_y2 = int(box2[0][1]), int(box2[0][3])

                    bb1 = {'x1': b1_x1, 'x2': b1_x2, 'y1': b1_y1, 'y2': b1_y2}
                    bb2 = {'x1': b2_x1, 'x2': b2_x2, 'y1': b2_y1, 'y2': b2_y2}
                    iou = get_iou(bb1, bb2, labelPath, predPath)
                    iouList.append(iou)
                    ######################################################
                    b1_x1, b1_x2 = int(w * (box1[1][1])) - int(w * (box1[1][3]) / 2), int(w * (box1[1][1])) + int(
                        w * (box1[1][3]) / 2)
                    b1_y1, b1_y2 = int(h * (box1[1][2])) - int(h * (box1[1][4]) / 2), int(h * (box1[1][2])) + int(
                        h * (box1[1][4]) / 2)
                    b2_x1, b2_x2 = int(box2[1][0]), int(box2[1][2])
                    b2_y1, b2_y2 = int(box2[1][1]), int(box2[1][3])

                    bb1 = {'x1': b1_x1, 'x2': b1_x2, 'y1': b1_y1, 'y2': b1_y2}
                    bb2 = {'x1': b2_x1, 'x2': b2_x2, 'y1': b2_y1, 'y2': b2_y2}
                    iou = get_iou(bb1, bb2, labelPath, predPath)
                    iouList.append(iou)

            else:
                b1_x1, b1_x2 = int(w * (box1[0][1])) - int(w * (box1[0][3]) / 2), int(w * (box1[0][1])) + int(
                    w * (box1[0][3]) / 2)
                b1_y1, b1_y2 = int(h * (box1[0][2])) - int(h * (box1[0][4]) / 2), int(h * (box1[0][2])) + int(
                    h * (box1[0][4]) / 2)
                b2_x1, b2_x2 = int(box2[0][0]), int(box2[0][2])
                b2_y1, b2_y2 = int(box2[0][1]), int(box2[0][3])

                bb1 = {'x1': b1_x1, 'x2': b1_x2, 'y1': b1_y1, 'y2': b1_y2}
                bb2 = {'x1': b2_x1, 'x2': b2_x2, 'y1': b2_y1, 'y2': b2_y2}
                iou = get_iou(bb1, bb2, labelPath, predPath)
                iouList.append(iou)
        else:
            iouList.append(0)

    return iouList




iouList = iou(path1, path2)

print(mean(iouList))
shutil.copy(jpgPath, path3)

import torch

torch.manual_seed(218)
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2
import math
import statistics
import pandas as pd

class NestCV(object):
    def __init__(self, dataset, nestCVTestIdx, nestCVFoldIdx):
        self.dataset = dataset
        self.nestCVTestIdx = nestCVTestIdx
        self.nestCVFoldIdx = nestCVFoldIdx
        
    def subCV(self, trainAndValIndices, trainAndValLen):
        if self.nestCVFoldIdx == 1:
            valIndices = trainAndValIndices[:int(trainAndValLen * 0.2)]
            trainIndices = trainAndValIndices[int(trainAndValLen * 0.2):]
        if self.nestCVFoldIdx == 2:
            valIndices = trainAndValIndices[int(trainAndValLen * 0.2):int(trainAndValLen * 0.4)]
            trainIndices = trainAndValIndices[:int(trainAndValLen * 0.2)] + \
                           trainAndValIndices[int(trainAndValLen * 0.4):]
        if self.nestCVFoldIdx == 3:
            valIndices = trainAndValIndices[int(trainAndValLen * 0.4):int(trainAndValLen * 0.6)]
            trainIndices = trainAndValIndices[:int(trainAndValLen * 0.4)] + \
                           trainAndValIndices[int(trainAndValLen * 0.6):]
        if self.nestCVFoldIdx == 4:
            valIndices = trainAndValIndices[int(trainAndValLen * 0.6):int(trainAndValLen * 0.8)]
            trainIndices = trainAndValIndices[:int(trainAndValLen * 0.6)] + \
                           trainAndValIndices[int(trainAndValLen * 0.8):]
        if self.nestCVFoldIdx == 5:
            valIndices = trainAndValIndices[int(trainAndValLen * 0.8):]
            trainIndices = trainAndValIndices[:int(trainAndValLen * 0.8)]
        return valIndices, trainIndices

    def nestCV(self):
        
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        
        if self.nestCVTestIdx == 1:
            testIndices = indices[:int(dataset_size * 0.2)]
            trainAndValIndices = indices[int(dataset_size * 0.2):]
            trainAndValLen = len(trainAndValIndices)
            valIndices,trainIndices = self.subCV(trainAndValIndices, trainAndValLen)
        if self.nestCVTestIdx == 2:
            testIndices = indices[int(dataset_size * 0.2):int(dataset_size * 0.4)]
            trainAndValIndices = indices[:int(dataset_size * 0.2)]+indices[int(dataset_size * 0.4):]
            trainAndValLen = len(trainAndValIndices)
            valIndices,trainIndices = self.subCV(trainAndValIndices, trainAndValLen)
        if self.nestCVTestIdx == 3:
            testIndices = indices[int(dataset_size * 0.4):int(dataset_size * 0.6)]
            trainAndValIndices = indices[:int(dataset_size * 0.6)]+indices[int(dataset_size * 0.6):]
            trainAndValLen = len(trainAndValIndices)
            valIndices,trainIndices = self.subCV(trainAndValIndices, trainAndValLen)
        if self.nestCVTestIdx == 4:
            testIndices = indices[int(dataset_size * 0.6):int(dataset_size * 0.8)]
            trainAndValIndices = indices[:int(dataset_size * 0.6)]+indices[int(dataset_size * 0.8):]
            trainAndValLen = len(trainAndValIndices)
            valIndices,trainIndices = self.subCV(trainAndValIndices, trainAndValLen)
        
        if self.nestCVTestIdx == 5:
            testIndices = indices[int(dataset_size * 0.8):]
            trainAndValIndices = indices[:int(dataset_size * 0.8)]
            trainAndValLen = len(trainAndValIndices)
            valIndices,trainIndices = self.subCV(trainAndValIndices, trainAndValLen)
        return trainIndices, valIndices, testIndices
    
import torch
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
from skimage import io# Ignore warnings
import warnings
import math
from skimage.transform import resize
warnings.filterwarnings("ignore")

class HeatmapDataset(Dataset):
    def __init__(self, csvFile, rootDir, resolution, flag, transform=None, prediction=0, originalUNet= True, deepWiseUNet=False, deepWiseUNetTwoInput=False, twoInputUNet=False):

        self.dataframe = pd.read_csv(csvFile)
        self.rootDir = rootDir
        self.transform = transform
        self.prediction = prediction
        self.resolution = int(resolution)
        self.flag = flag
        self.originalUNet = originalUNet
        self.deepWiseUNet = deepWiseUNet
        self.deepWiseUNetTwoInput = deepWiseUNetTwoInput
        self.twoInputUNet = twoInputUNet



    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.originalUNet:
            imgPath = os.path.join(self.rootDir,'data/'+self.flag +'/image/',
                                   self.dataframe.iloc[idx, 0])
            if not os.path.exists(imgPath):
                print(imgPath)

            image = io.imread(imgPath, as_gray=True).reshape(1,512,512)
            if self.prediction == 0:
                heatmapList = []
                for heatmapIndex in range(1,2):
                    heatmapName = self.dataframe.iloc[idx, heatmapIndex]
                    heatmapArr = io.imread(self.rootDir+'data/'+self.flag+'/label/'+heatmapName, as_gray=True)
                    heatmapList.append(heatmapArr)

                label =heatmapList[0].reshape(1,512,512) #np.dstack((heatmapList[0],heatmapList[1],heatmapList[2]))
                # label = np.moveaxis(label, -1, 0)
                # print('label:',label.shape)
                sample = {'image': torch.Tensor(image),'label': torch.Tensor(label), 'imageName': self.dataframe.iloc[idx, 0]}
                if self.transform:
                    sample = self.transform(sample)
                return sample
            else:
                sample = {'image': torch.Tensor(image), 'imageName': self.dataframe.iloc[idx, 0]}
                if self.transform:
                    sample = self.transform(sample)
                return sample

        if self.deepWiseUNet:
            imgPath = os.path.join(self.rootDir, 'data/' + self.flag + '/image/',
                                   self.dataframe.iloc[idx, 0])
            if not os.path.exists(imgPath):
                print(imgPath)

            image = io.imread(imgPath, as_gray=True).reshape(1, 512, 512)
            if self.prediction == 0:

                heatmapList = []
                for heatmapIndex in range(1, 2):
                    heatmapName = self.dataframe.iloc[idx, heatmapIndex]
                    heatmapArr = io.imread(self.rootDir + 'data/' + self.flag + '/label/' + heatmapName, as_gray=True)
                    heatmapList.append(heatmapArr)

                label = heatmapList[0].reshape(1, 512, 512)  # np.dstack((heatmapList[0],heatmapList[1],heatmapList[2]))
                # label = np.moveaxis(label, -1, 0)
                # print('label:',label.shape)
                sample = {'image': torch.Tensor(image), 'label': torch.Tensor(label),
                          'imageName': self.dataframe.iloc[idx, 0]}
                if self.transform:
                    sample = self.transform(sample)
                return sample
            else:
                sample = {'image': torch.Tensor(image), 'imageName': self.dataframe.iloc[idx, 0]}
                if self.transform:
                    sample = self.transform(sample)
                return sample

        if self.deepWiseUNetTwoInput or self.twoInputUNet:
            imgPath = os.path.join(self.rootDir, 'data/' + self.flag + '/image/',
                                   self.dataframe.iloc[idx, 0])
            if not os.path.exists(imgPath):
                print(imgPath)

            image = io.imread(imgPath, as_gray=True).reshape(1, 512, 512)
            imageStage2 = resize(io.imread(imgPath, as_gray=True), (256,256)).reshape(1,256,256)

            if self.prediction == 0:
                heatmapList = []
                for heatmapIndex in range(1, 2):
                    heatmapName = self.dataframe.iloc[idx, heatmapIndex]
                    heatmapArr = io.imread(self.rootDir + 'data/' + self.flag + '/label/' + heatmapName, as_gray=True)
                    heatmapList.append(heatmapArr)

                label = heatmapList[0].reshape(1, 512, 512)  # np.dstack((heatmapList[0],heatmapList[1],heatmapList[2]))

                sample = {'image': torch.Tensor(image), 'imageStage2':torch.Tensor(imageStage2), 'label': torch.Tensor(label),
                          'imageName': self.dataframe.iloc[idx, 0]}
                if self.transform:
                    sample = self.transform(sample)
                return sample
            else:
                sample = {'image': torch.Tensor(image), 'imageStage2':imageStage2,'imageName': self.dataframe.iloc[idx, 0]}
                if self.transform:
                    sample = self.transform(sample)
                return sample


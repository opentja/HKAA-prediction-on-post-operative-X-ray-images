from customDataset import HeatmapDataset

import torch

torch.manual_seed(218)
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
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
from nestCV import NestCV
import math
from scipy.stats import gmean

class TrainTestPrediction(object):
    def __init__(self, model, device, args):
        self.batchSize = args.batchSize
        self.epochs = args.epochs
        self.lr = args.lr
        self.device = device
        if args.originalUNet == True:
            modelName = 'originalUNet'
        elif args.deepWiseUNet == True:
            modelName = 'deepWiseUNet'
        elif args.deepWiseUNetTwoInput ==True:
            modelName = 'deepWiseUNetTwoInput'
        else:
            modelName = 'twoInputUNet'

        self.rootDir = args.rootDir + str(args.sigma)+'/'
        self.resolution = args.resolution
        self.flag=args.flag
        self.sigma = args.sigma
        self.annotation = self.rootDir +'data/'+ args.flag+ '/newannotation.csv'
        self.originalUNet = args.originalUNet
        self.deepWiseUNet = args.deepWiseUNet
        self.deepWiseUNetTwoInput = args.deepWiseUNetTwoInput
        self.twoInputUNet = args.twoInputUNet
        self.nestCVTestIdx = args.nestCVTestIdx
        self.nestCVFoldIdx = args.nestCVFoldIdx
        self.CV = args.CV

        if args.prediction == 0:
            dataset = HeatmapDataset(csvFile=self.annotation, rootDir=self.rootDir, resolution=self.resolution,
                                     flag=args.flag, transform=None,prediction=0,
                                     originalUNet=self.originalUNet,
                                     deepWiseUNet=self.deepWiseUNet,
                                     deepWiseUNetTwoInput=self.deepWiseUNetTwoInput,
                                     twoInputUNet = self.twoInputUNet)
            if self.CV == True:
                nestCVObj = NestCV(dataset, self.nestCVTestIdx, self.nestCVFoldIdx)
                trainIndices, validIndices, testIndices = nestCVObj.nestCV()
                self.trainSet = Subset(dataset, trainIndices)
                self.validSet = Subset(dataset, validIndices)
                self.testSet = Subset(dataset, testIndices)

            else:
                trainNumber = int(len(dataset) * 0.6)
                validNumber = int(len(dataset) * 0.2)
                testNumber = len(dataset) - trainNumber - validNumber
                self.trainSet, self.validSet, self.testSet = random_split(dataset, [trainNumber, validNumber, testNumber])

        else:
            annotationPath = 'kneeX-ray/predictionBasedYoloOuput/unAnnotatedImage/data/'+args.flag+'/annotation.csv'
            predImageFolder = 'kneeX-ray/predictionBasedYoloOuput/unAnnotatedImage/'
            dataset = HeatmapDataset(csvFile=annotationPath, rootDir=predImageFolder, resolution=self.resolution,
                                    flag=args.flag, transform=None,prediction=1,
                                     originalUNet=self.originalUNet,
                                     deepWiseUNet=self.deepWiseUNet,
                                     deepWiseUNetTwoInput=self.deepWiseUNetTwoInput,
                                     twoInputUNet = self.twoInputUNet)
            self.predImageFolder = 'kneeX-ray/predictionBasedYoloOuput/unAnnotatedImage/data/'+args.flag+'/image/'
            self.predLoader = DataLoader(dataset=dataset, batch_size=self.batchSize, shuffle=True)
            self.predictionResultPath = predImageFolder +'/'+ +modelName+'/experimentResult/' + self.flag + '/prediction/'
            if not os.path.exists(self.predictionResultPath+'image/'):
                os.makedirs(self.predictionResultPath+'image/')
        self.model = model


        self.trainResultPath = self.rootDir+modelName+'/'+'experimentResult/'+self.flag+'/train'\
                               +str(self.nestCVTestIdx)+'/fold'+str(self.nestCVTestIdx)+'/'
        self.trainResultPath = self.rootDir  + 'experimentResult/' + self.flag + '/train/'

        if not os.path.exists(self.trainResultPath + 'model/'):
            os.makedirs(self.trainResultPath + 'model/')
        if not os.path.exists(self.trainResultPath + 'curve/'):
            os.makedirs(self.trainResultPath + 'curve/')
        # if not os.path.exists(self.trainResultPath + 'image/'):
        #     os.makedirs(self.trainResultPath + 'image/')

        self.testResultPath = self.rootDir+modelName+'/'+'experimentResult/'+self.flag+'/test'\
                              +str(self.nestCVTestIdx) +'/'
        self.testResultPath = self.rootDir + 'experimentResult/' +self.flag+'/test'

        if not os.path.exists(self.testResultPath + 'image/'):
            os.makedirs(self.testResultPath + 'image/')

    def getXYFromHeatmapsArr(self, heatmapsArr):
        landmarksList = []
        for i in range(1):
            heatmap = heatmapsArr[i, :, :]
            a = heatmap.argsort(axis=None)[-25:]
            # print('heatmap shape:', heatmap.shape)
            topind = np.unravel_index(a, heatmap.shape)
            y_list = list(topind[0])
            x_list = list(topind[1])

            # topind (array1, array2)
            # array1 represent  which row
            # array2 represent which col
            # heatmap [0,3] ==>> row 0, col 3
            ######weighted coordinate########
            # i0, i1, hsum = 0, 0, 0
            # for ind in zip(topind[0], topind[1]):
            #     h = heatmap[ind[0], ind[1]]
            #     hsum += h
            #     i0 += ind[0] * h
            #     i1 += ind[1] * h
            #
            # i0 /= hsum
            # i1 /= hsum
            #
            # i0 = int(round(i0, 0))
            # i1 = int(round(i1, 0))

            # ######harmonic_mean########
            # i1 = int(statistics.harmonic_mean(x_list))
            # i0 = int(statistics.harmonic_mean(y_list))

            ######geometric_mean########
            i1 = int(gmean(x_list))
            i0 = int(gmean(y_list))

            landmarksList.append(i1) # X coordinate
            landmarksList.append(i0) # Y coordinate
        # print(len(landmarksList))
        return landmarksList

    def train(self):
        trainDataloader = DataLoader(dataset=self.trainSet, batch_size=self.batchSize, shuffle=True)
        validDataloader = DataLoader(dataset=self.validSet, batch_size=self.batchSize, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        validLossFlag = 999999999

        # Train the model
        trainLossList = []
        validLossList = []
        trainingStart = time.time()

        for epoch in range(self.epochs):
            epochStart = time.time()
            # if epoch > 48:
            #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr * 0.1)
            print('Best Validation Loss: ', validLossFlag)
            trainLosses = []
            validLosses = []
            self.model.train()
            for i, sample_batched in enumerate(trainDataloader, 1):
                if self.deepWiseUNetTwoInput == False and self.twoInputUNet == False:
                    image = sample_batched['image'].to(self.device)
                    heatmaps = sample_batched['label'].to(self.device)
                    optimizer.zero_grad()
                    output = self.model(image)
                else:
                    image = sample_batched['image'].to(self.device)
                    imageStage2 = sample_batched['imageStage2'].to(self.device)
                    heatmaps = sample_batched['label'].to(self.device)
                    optimizer.zero_grad()
                    output = self.model(image, imageStage2)
                loss = criterion(output, heatmaps)
                loss.backward()
                optimizer.step()
                #
            # recaculate train loss in model.eval()
            self.model.eval()  # prep model for evaluation
            for sample_batched in trainDataloader:
                if self.deepWiseUNetTwoInput == False  and self.twoInputUNet == False:
                    image = sample_batched['image'].to(self.device)
                    heatmaps = sample_batched['label'].to(self.device)
                    output = self.model(image)
                else:
                    image = sample_batched['image'].to(self.device)
                    imageStage2 = sample_batched['imageStage2'].to(self.device)
                    heatmaps = sample_batched['label'].to(self.device)
                    optimizer.zero_grad()
                    output = self.model(image, imageStage2)
                # print(output.shape)
                loss = criterion(output, heatmaps)
                trainLosses.append(loss.item())

            # validate the model #
            counter = 0
            for sample_batched in validDataloader:
                counter += 1
                if self.deepWiseUNetTwoInput == False and self.twoInputUNet == False:
                    image = sample_batched['image'].to(self.device)
                    heatmaps = sample_batched['label'].to(self.device)
                    output = self.model(image)
                else:
                    image = sample_batched['image'].to(self.device)
                    imageStage2 = sample_batched['imageStage2'].to(self.device)
                    heatmaps = sample_batched['label'].to(self.device)
                    output = self.model(image, imageStage2)

                loss = criterion(output, heatmaps)
                validLosses.append(loss.item())

            trainLoss = np.average(trainLosses)
            validLoss = np.average(validLosses)
            trainLossList.append(10000 * trainLoss)
            validLossList.append(10000 * validLoss)

            if validLoss < validLossFlag:
                validLossFlag = round(validLoss, 6)
                torch.save(self.model.state_dict(),
                           self.trainResultPath + 'model/bestModelWithEpochs' + str(self.epochs) + 'Lr'
                           + str(self.lr) + 'BatchSize' + str(self.batchSize) + 'Resolution' + str(
                               self.resolution) + '.pt')

            if len(trainLossList) > 10:
                xRange = np.arange(10, epoch) + 1
                plt.plot(xRange, trainLossList[11:], 'b', label='Training loss')
                plt.plot(xRange, validLossList[11:], 'r', label='Validation loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss(Per image) ')
                plt.legend()
                plt.show()
                lossRecordImageName = 'lossRecordOfEpochs' + str(
                    self.epochs) + 'Lr' + str(self.lr) + 'BatchSize' + str(self.batchSize) + 'Resolution' + str(
                    self.resolution) + '.jpg'

                plt.savefig(self.trainResultPath + 'curve/' + lossRecordImageName)
                plt.close()
            epochEnd = time.time()

            resultTxt = open(
                self.trainResultPath + 'Epochs' + str(self.epochs) + 'Lr' + str(self.lr) + 'BatchSize' + str(
                    self.batchSize) + 'Resolution' + str(self.resolution) + '.txt', 'a+')
            resultTxt.write('epoch:' + str(epoch) + '    trainLoss: ' + str(round(10000 * trainLoss, 6)) +
                            '    validationLoss: ' + str(round(10000 * validLoss, 6)) + '    timeCost:  ' + str(
                epochEnd - epochStart) + '\n')
            resultTxt.close()

            print('epoch:', epoch, 'trainLoss: ', round(10000 * trainLoss, 6),
                  ' validationLoss: ', round(10000 * validLoss, 6), '      timeCost:  ', epochEnd - epochStart)

        trainingEnd = time.time()
        resultTxt = open(self.trainResultPath + 'Epochs' + str(self.epochs) + 'Lr' + str(self.lr) + 'BatchSize' + str(
            self.batchSize) + 'Resolution' + str(self.resolution) + '.txt', 'a+')

        resultTxt.write('best validation loss:  ' + str(validLossFlag) + '\n')
        resultTxt.write('traing time cost in total:  ' + str(trainingEnd - trainingStart) + '\n')
        resultTxt.close()
        print('traing time cost in total: ', trainingEnd - trainingStart)

    def calculateAngle(self, hipCoor, kneeCoor, ankleCoor, leftOrRight):
        ang = math.degrees(
            math.atan2(ankleCoor[1] - kneeCoor[1], ankleCoor[0] - kneeCoor[0]) - math.atan2(hipCoor[1] - kneeCoor[1],
                                                                                            hipCoor[0] - kneeCoor[0]))
        # print(ang)
        if leftOrRight == 'left':
            return ang - 180
        if leftOrRight == 'right':
            return -(ang - 180)
    def calculateDistance(self,pred,label):
        x1, y1 = pred[0], pred[1]
        x2, y2 = label[0], label[1]
        distance = round(math.sqrt((x1-x2)*(x1-x2)+(y2-y1)*(y2-y1)), 2)
        return distance


    def test(self):
        testDataloader = DataLoader(dataset=self.testSet, batch_size=self.batchSize, shuffle=True)
        testLosses, imageNameAndCenterLocationList = [], []
        stateDict = torch.load(self.trainResultPath + 'model/bestModelWithEpochs' + str(self.epochs) + 'Lr'
                           + str(self.lr) + 'BatchSize' + str(self.batchSize) + 'Resolution' + str(
                               self.resolution) + '.pt')
        # newStateDict = OrderedDict()
        # for k, v in stateDict.items():
        #     name = k[7:]  # remove module
        #     newStateDict[name] = v
        self.model.load_state_dict(stateDict)
        criterion = nn.MSELoss()

        self.model.eval()
        imageNameList, predXList, predYList, labelXList, labelYList, distanceList = [], [],[],[],[],[]
        for sample_batched in testDataloader:
            if self.deepWiseUNetTwoInput == False  and self.twoInputUNet == False:
                image = sample_batched['image'].to(self.device)
                heatmaps = sample_batched['label'].to(self.device)
                output = self.model(image)
            else:
                image = sample_batched['image'].to(self.device)
                imageStage2 = sample_batched['imageStage2'].to(self.device)
                heatmaps = sample_batched['label'].to(self.device)
                output = self.model(image, imageStage2)
            loss = criterion(output, heatmaps)
            testLosses.append(loss.item())

            outputHeatmapArray = output.cpu().detach().numpy()
            labelHeatmapArray = heatmaps.cpu().detach().numpy()
            imageNameBatch = sample_batched['imageName']

            for idx in range(outputHeatmapArray.shape[0]):
                imageName = imageNameBatch[idx]
                imagePath = self.rootDir+'data/'+self.flag+ '/image/'+imageName
                imageArray = cv2.imread(imagePath)
                # print(labelHeatmapArray[idx].shape)

                labelList = self.getXYFromHeatmapsArr(labelHeatmapArray[idx])
                predictionList = self.getXYFromHeatmapsArr(outputHeatmapArray[idx])

                pointRadius = 3
                redColor = (0, 0, 255)
                greenColor = (0, 255, 0)
                blueColor = (255, 0, 0)
                thickness = -1
                # print(labelList)
                # print(predictionList,'\n')
                # mark center of label: blue
                cv2.circle(imageArray, (labelList[0], labelList[1]), pointRadius, blueColor, thickness)

                # mark center of prediction: red
                cv2.circle(imageArray, (predictionList[0], predictionList[1]), pointRadius, redColor, thickness)
                plt.subplot(111)
                plt.imshow(outputHeatmapArray[idx][0, :, :])

                plt.savefig(self.testResultPath + 'image/' + imageName.replace('.jpg', '') + 'predictedHeatmap.jpg')
                plt.close()

                # save image with prediction
                cv2.imwrite(self.testResultPath + 'image/' + imageName.replace('.jpg', '') + 'recaledImage.jpg', imageArray)
                # save coordinate into txt file
                distance = self.calculateDistance(predictionList, labelList)
                distanceList.append(distance)
                predXList.append(predictionList[0])
                predYList.append(predictionList[1])
                labelXList.append(labelList[0])
                labelYList.append(labelList[1])
                imageNameList.append(imageName)
        print('average distance of coordinates is: ', round(sum(distanceList)/len(distanceList),2))
        # save test result as csv file
        predInfo = {'imageName': imageNameList, 'labelX':labelXList, 'labelY':labelYList, 'predX': predXList,
                    'predY':predYList, 'distance':distanceList}


        df = pd.DataFrame(predInfo)
        df.to_csv(self.testResultPath + '/'+self.flag+'PredInfo.csv', index=False)
        testLossAvg = np.average(testLosses)
        print('average test loss is: ', testLossAvg)
        print('average distance difference is: ', np.average(distanceList))


    def pred(self):
        predDataloader = self.predLoader
        stateDict = torch.load(self.trainResultPath + 'model/bestModelWithEpochs' + str(self.epochs) + 'Lr'
                           + str(self.lr) + 'BatchSize' + str(self.batchSize) + 'Resolution' + str(
                               self.resolution) + '.pt')
        self.model.load_state_dict(stateDict)

        self.model.eval()
        imageNameList, predXList, predYList = [], [], []
        for sample_batched in predDataloader:
            if self.deepWiseUNetTwoInput ==False  and self.twoInputUNet == False:
                image = sample_batched['image'].to(self.device)
                output = self.model(image)
            else:
                image = sample_batched['image'].to(self.device)
                imageStage2 = sample_batched['imageStage2'].to(self.device)
                output = self.model(image, imageStage2)

            outputHeatmapArray = output.cpu().detach().numpy()
            imageNameBatch = sample_batched['imageName']

            for idx in range(outputHeatmapArray.shape[0]):
                imageName = imageNameBatch[idx]
                imagePath = self.predImageFolder+imageName
                # print(imageName)
                # print(self.predImageFolder)
                # print(os.path.exists(imagePath))
                imageArray = cv2.imread(imagePath)
                # print('imageArr', imageArray.shape)
                predictionList = self.getXYFromHeatmapsArr(outputHeatmapArray[idx])

                pointRadius = 3
                redColor = (0, 0, 255)
                thickness = -1
                # print(labelList)
                # print(predictionList,'\n')
                # mark center of prediction: red
                cv2.circle(imageArray, (predictionList[0], predictionList[1]), pointRadius, redColor, thickness)
                plt.subplot(111)
                plt.imshow(outputHeatmapArray[idx][0, :, :])

                plt.savefig(self.predictionResultPath + 'image/' + imageName.replace('.jpg', '') + 'predictedHeatmap.jpg')
                plt.close()

                # save image with prediction
                cv2.imwrite(self.predictionResultPath + 'image/' + imageName.replace('.jpg', '') + 'recaledImage.jpg', imageArray)
                # save coordinate into txt file
                predXList.append(predictionList[0])
                predYList.append(predictionList[1])
                imageNameList.append(imageName)
        print(len(imageNameList))
        print(len(predXList))
        print(len(predYList))
        # save prediction result as csv file
        predInfo = {'imageName': imageNameList, 'predX': predXList,'predY':predYList}
        df = pd.DataFrame(predInfo)
        df.to_csv(self.predictionResultPath + '/predInfo.csv', index=False)



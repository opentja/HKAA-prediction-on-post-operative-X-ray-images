# reference https://github.com/ultralytics/yolov3

import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def calculateAvg(results1, results2, results3, results4, results5, index, x):
    y1 = results1[index, x].tolist()
    y2 = results2[index, x].tolist()
    y3 = results3[index, x].tolist()
    y4 = results4[index, x].tolist()
    y5 = results5[index, x].tolist()
    y = []
    for i in range(len(y1)):
        y.append(round((y1[i] + y2[i] + y3[i] + y4[i] + y5[i]) / 5, 3))
    return y


def calculateAvg2(res11, res21, res31, res41, res51, index, x):
    y11 = res11[index, x].tolist()
    y21 = res21[index, x].tolist()
    y31 = res31[index, x].tolist()
    y41 = res41[index, x].tolist()
    y51 = res51[index, x].tolist()

    y = []
    for i in range(len(y11)):
        y.append(
            round((y11[i] + y21[i] + y31[i] + y41[i] + y51[i]) / 5, 3))
    return y


def readTxt(resultTxtPath, epochs, batchSize, lr):
    # for each hyperparameter combination, calculate the average performance of 5 epochs.
    experimentResultDict = {}
    # use dict to save loss and accuracy{epoch64: [], epoch128:[],epoch256:[]}
    # [trainGLoss, trainObjLoss, trainClsLoss, valGloss, valObjLoss, valClsLoss, precision, recall, meanap, f1]
    path11 = resultTxtPath + '1/batchSize' + str(batchSize) + 'Epochs' + str(epochs) + 'LR' + str(
        lr) + '-0results.txt'

    path21 = resultTxtPath + '2/batchSize' + str(batchSize) + 'Epochs' + str(epochs) + 'LR' + str(
        lr) + '-0results.txt'

    path31 = resultTxtPath + '3/batchSize' + str(batchSize) + 'Epochs' + str(epochs) + 'LR' + str(
        lr) + '-0results.txt'

    path41 = resultTxtPath + '4/batchSize' + str(batchSize) + 'Epochs' + str(epochs) + 'LR' + str(
        lr) + '-0results.txt'

    path51 = resultTxtPath + '5/batchSize' + str(batchSize) + 'Epochs' + str(epochs) + 'LR' + str(
        lr) + '-0results.txt'

    res11 = np.loadtxt(path11, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    res21 = np.loadtxt(path21, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    res31 = np.loadtxt(path31, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    res41 = np.loadtxt(path41, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    res51 = np.loadtxt(path51, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T

    n = res11.shape[1]  # number of rows
    x = range(1, n)

    trainGiouLoss = calculateAvg2(res11, res21,  res31,  res41,  res51,  0, x)
    trainObjectivenessLoss = calculateAvg2(res11,  res21,  res31,  res41,  res51,  1, x)
    trainClassificationLoss = calculateAvg2(res11,  res21,  res31,  res41,  res51,  2, x)

    valGiouLoss = calculateAvg2(res11,  res21,  res31,  res41,  res51,  5, x)
    valObjectivenessLoss = calculateAvg2(res11,  res21,  res31,  res41,  res51,  6, x)
    valClassificationLoss = calculateAvg2(res11,  res21,  res31,  res41,  res51,  7, x)

    precision = calculateAvg2(res11,  res21,  res31,  res41,  res51,  3, x)
    recall = calculateAvg2(res11,  res21,  res31,  res41,  res51,  4, x)
    meanAP = calculateAvg2(res11,  res21,  res31,  res41,  res51,  8, x)
    f1 = calculateAvg2(res11,  res21,  res31,  res41,  res51,  9, x)
    tmpList = [trainGiouLoss, trainObjectivenessLoss, trainClassificationLoss, valGiouLoss,
               valObjectivenessLoss, valClassificationLoss, precision, recall, meanAP, f1]
    return tmpList


def plot(d, epoch, lr, resultPath):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ax = ax.ravel()

    s = ['(a) GIoU', '(b) Objectness', '(c) Classification',
         '(d) val GIoU', '(e) val Objectness', '(f) val Classification']
    n = len(d[0])
    x = range(1, n)

    ax[0].plot(x, d[0][1:], color='green', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[0].set_title(s[0])
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')
    ax[0].legend()

    ax[1].plot(x, d[1][1:], color='green', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[1].set_title(s[1])
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    ax[1].legend()

    ax[2].plot(x, d[2][1:], color='green', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[2].set_title(s[2])
    ax[2].set_xlabel('epochs')
    ax[2].set_ylabel('loss')
    ax[2].legend()

    ax[3].plot(x, d[3][1:], color='blue', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[3].set_title(s[3])
    ax[3].set_xlabel('epochs')
    ax[3].set_ylabel('loss')
    ax[3].legend()

    ax[4].plot(x, d[4][1:], color='blue', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[4].set_title(s[4])
    ax[4].set_xlabel('epochs')
    ax[4].set_ylabel('loss')
    ax[4].legend()

    ax[5].plot(x, d[5][1:], color='blue', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[5].set_title(s[5])
    ax[5].set_xlabel('epochs')
    ax[5].set_ylabel('loss')
    ax[5].legend()

    fig.tight_layout()
    fig.savefig(resultPath + 'averageLossCurveOfEpoch' + str(epoch) + 'lr' + str(lr) + '.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.ravel()
    s = ['(a) Precision', '(b) mAP@0.5', '(c) Recall', '(d) F1']

    ax[0].plot(x, d[6][1:], color='blue', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[0].set_title(s[0])
    ax[0].set_xlabel('epochs')
    ax[0].legend()

    ax[1].plot(x, d[7][1:], color='blue', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[1].set_title(s[1])
    ax[1].set_xlabel('epochs')
    ax[1].legend()

    ax[2].plot(x, d[8][1:], color='blue', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[2].set_title(s[2])
    ax[2].set_xlabel('epochs')
    ax[2].legend()

    ax[3].plot(x, d[9][1:], color='blue', marker='.', label='epoch' + str(epoch) + 'batchSize64lr' + str(lr))
    ax[3].set_title(s[3])
    ax[3].set_xlabel('epochs')
    ax[3].legend()

    fig.tight_layout()
    fig.savefig(resultPath + 'averageAccuracyCurveOfEpoch' + str(epoch) + 'lr' + str(lr) + '.png', dpi=200)
    plt.rcParams.update({'axes.titlesize': 'large'})
    plt.close(fig)


def getAllResStat(resultTxtPath, resultPath):
    # learning rate 0.001

    epoch64Batch64LR001 = readTxt(resultTxtPath, 512, 64, 0.0001)

    plot(epoch64Batch64LR001, 512, 0.0001, resultPath)





def plot_results(start=1, stop=0, id=(), resultPath='', resultTxtPath=''):  # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    resultList = []
    # batchSize64Epochs256LR0.0001-2results.txt
    for idx in range(0, 4):
        if idx == 0:
            batchSize = 8
        if idx == 1:
            batchSize = 16
        if idx == 2:
            batchSize = 32
        if idx == 3:
            batchSize = 64

    results1 = np.loadtxt(resultTxtPath + '1/batchSize32Epochs64LR0.001results.txt',
                          usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    results2 = np.loadtxt(resultTxtPath + '2/batchSize32Epochs64LR0.001results.txt',
                          usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    results3 = np.loadtxt(resultTxtPath + '3/batchSize32Epochs64LR0.001results.txt',
                          usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    results4 = np.loadtxt(resultTxtPath + '4/batchSize32Epochs64LR0.001results.txt',
                          usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    results5 = np.loadtxt(resultTxtPath + '5/batchSize32Epochs64LR0.001results.txt',
                          usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T

    n = results1.shape[1]  # number of rows

    x = range(1, n)

    # 0,1,2,5,6,7 is loss
    # 3,4,8,9 is accuracy

    trainGiouLoss = calculateAvg(results1, results2, results3, results4, results5, 0, x)
    trainObjectivenessLoss = calculateAvg(results1, results2, results3, results4, results5, 1, x)
    trainClassificationLoss = calculateAvg(results1, results2, results3, results4, results5, 2, x)

    valGiouLoss = calculateAvg(results1, results2, results3, results4, results5, 5, x)
    valObjectivenessLoss = calculateAvg(results1, results2, results3, results4, results5, 6, x)
    valClassificationLoss = calculateAvg(results1, results2, results3, results4, results5, 7, x)

    precision = calculateAvg(results1, results2, results3, results4, results5, 3, x)
    recall = calculateAvg(results1, results2, results3, results4, results5, 4, x)
    meanAP = calculateAvg(results1, results2, results3, results4, results5, 8, x)
    f1 = calculateAvg(results1, results2, results3, results4, results5, 9, x)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ax = ax.ravel()

    s = ['(a) GIoU', '(b) Objectness', '(c) Classification',
         '(d) val GIoU', '(e) val Objectness', '(f) val Classification']

    print(len(trainObjectivenessLoss))
    print(x)

    ax[0].plot(x, trainGiouLoss, color='green', marker='.', label='loss value')
    ax[0].set_title(s[0])
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')
    ax[0].legend()

    ax[1].plot(x, trainObjectivenessLoss, color='green', marker='.', label='loss value')
    ax[1].set_title(s[1])
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    ax[1].legend()

    ax[2].plot(x, trainClassificationLoss, color='green', marker='.', label='loss value')
    ax[2].set_title(s[2])
    ax[2].set_xlabel('epochs')
    ax[2].set_ylabel('loss')
    ax[2].legend()

    ax[3].plot(x, valGiouLoss, color='green', marker='.', label='loss value')
    ax[3].set_title(s[3])
    ax[3].set_xlabel('epochs')
    ax[3].set_ylabel('loss')
    ax[3].legend()

    ax[4].plot(x, valObjectivenessLoss, color='green', marker='.', label='loss value')
    ax[4].set_title(s[4])
    ax[4].set_xlabel('epochs')
    ax[4].set_ylabel('loss')
    ax[4].legend()

    ax[5].plot(x, valClassificationLoss, color='green', marker='.', label='loss value')
    ax[5].set_title(s[5])
    ax[5].set_xlabel('epochs')
    ax[5].set_ylabel('loss')
    ax[5].legend()

    fig.tight_layout()
    fig.savefig(resultPath + 'averageLossCruve64.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.ravel()

    s = ['(a) Precision', '(b) mAP@0.5', '(c) Recall', '(d) F1']
    # font = {'fontsize': rcParams['axes.titlesize'],'fontweight' : rcParams['axes.titleweight'],'verticalalignment': 'baseline','horizontalalignment': loc}

    ax[0].plot(x, precision, color='green', marker='.')
    ax[0].set_title(s[0])
    ax[0].set_xlabel('epochs')

    ax[1].plot(x, meanAP, color='green', marker='.')
    ax[1].set_title(s[1])
    ax[1].set_xlabel('epochs')

    ax[2].plot(x, recall, color='green', marker='.')
    ax[2].set_title(s[2])
    ax[2].set_xlabel('epochs')

    ax[3].plot(x, f1, color='green', marker='.')
    ax[3].set_title(s[3])
    ax[3].set_xlabel('epochs')

    fig.tight_layout()
    fig.savefig(resultPath + 'averageAccuracyCruve64.png', dpi=200)
    plt.rcParams.update({'axes.titlesize': 'large'})
    plt.close(fig)



# plot_results(resultPath, resultTxtPath=wdir)  # save as results.png
getAllResStat(resultTxtPath, resultPath)

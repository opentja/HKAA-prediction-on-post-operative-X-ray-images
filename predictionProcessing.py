import pandas as pd
import cv2
import os
import math


def predCoor(df):
    newPredXList, newPredYList = [], []
    for index, row in df.iterrows():
        ratio = row['ratio']
        predX = row['predX']
        predY = row['predY']
        leftZero = row['leftZeros']
        topZero = row['topZeros']
        xStart = row['x']
        yStart = row['y']
        newPredX = xStart + (int(predX * ratio) - leftZero)
        newPredY = yStart + (int(predY * ratio) - topZero)

        newPredXList.append(newPredX)
        newPredYList.append(newPredY)

    df['newPredX'] = newPredXList
    df['newPredY'] = newPredYList

    return df


def combineCSV(unetOutputCSVPath, roiCsvPath, annotationCsvPath, flag):
    unetOutputCSVPath = unetOutputCSVPath + flag + '/prediction/predInfo.csv'
    annotationCsvPath = annotationCsvPath + flag + '/annotation.csv'

    roiDF = pd.read_csv(roiCsvPath)
    unetDF = pd.read_csv(unetOutputCSVPath)
    annotationDF = pd.read_csv(annotationCsvPath)

    tmpDF = pd.merge(roiDF, unetDF, on='imageName')
    tmpDF2 = pd.merge(tmpDF, annotationDF, on='imageName')

    finalDF = predCoor(tmpDF2)
    # finalDF.to_csv('./'+flag+'combo.csv', index=False)
    # print(finalDF.shape)
    # print(finalDF.columns)
    return finalDF


def calculateAngle(hipCoor, kneeCoor, ankleCoor, leftOrRight):
    ang = math.degrees(
        math.atan2(ankleCoor[1] - kneeCoor[1], ankleCoor[0] - kneeCoor[0]) - math.atan2(hipCoor[1] - kneeCoor[1],
                                                                                        hipCoor[0] - kneeCoor[0]))
    # print(ang)
    if leftOrRight == 'left':
        return round(ang - 180, 2)
    if leftOrRight == 'right':
        return -round((ang - 180), 2)


def plotLandmark(hipDF, kneeDF, ankleDF, imageFolder, visualizationPath):
    if not os.path.exists(visualizationPath):
        os.makedirs(visualizationPath)
    imageNameList, predAngleList = [],[]
    leftAngleList, rightAngleList = [], []
    for index, row in hipDF.iterrows():
        imageNameWithoutLabel = row['imageNameListWithoutLabel']



        # print(index)
        # print(imageName)
        hipPredX, hipPredY = row['newPredX'], row['newPredY']
        kneePredX, kneePredY, kneeSide = kneeDF.iloc[index, 10], kneeDF.iloc[index, 11], kneeDF.iloc[index, 2]
        anklePredX, anklePredY = ankleDF.iloc[index, 10], ankleDF.iloc[index, 11]

        if not os.path.exists(visualizationPath + imageNameWithoutLabel):
            imageArray = cv2.imread(imageFolder + imageNameWithoutLabel)
        else:
            imageArray = cv2.imread(visualizationPath + imageNameWithoutLabel)

        pointRadius = 10
        redColor = (0, 0, 255)
        greenColor = (0, 255, 0)
        thickness = -1

        # mark center of prediction, red
        cv2.circle(imageArray, (hipPredX, hipPredY), pointRadius, redColor, thickness)
        cv2.circle(imageArray, (kneePredX, kneePredY), pointRadius, redColor, thickness)
        cv2.circle(imageArray, (anklePredX, anklePredY), pointRadius, redColor, thickness)

        # draw line
        # draw blue line among predicted points
        cv2.line(imageArray, (hipPredX, hipPredY), (kneePredX, kneePredY), greenColor, 3)
        cv2.line(imageArray, (kneePredX, kneePredY), (anklePredX, anklePredY), greenColor, 3)

        if kneeSide == 3 and imageNameWithoutLabel not in imageNameList:
            side = 'left'
            predAngle = calculateAngle((hipPredX, hipPredY), (kneePredX, kneePredY), (anklePredX, anklePredY), side)
            leftAngleList.append(predAngle)
            rightAngleList.append('NA')
            imageNameList.append(imageNameWithoutLabel)
        elif kneeSide == 3 and imageNameWithoutLabel in imageNameList:
            side = 'left'
            predAngle = calculateAngle((hipPredX, hipPredY), (kneePredX, kneePredY), (anklePredX, anklePredY), side)
            index = imageNameList.index(imageNameWithoutLabel)
            leftAngleList[index] = predAngle
        elif kneeSide == 7 and imageNameWithoutLabel not in imageNameList:
            side = 'right'
            predAngle = calculateAngle((hipPredX, hipPredY), (kneePredX, kneePredY), (anklePredX, anklePredY), side)
            leftAngleList.append('NA')
            rightAngleList.append(predAngle)
            imageNameList.append(imageNameWithoutLabel)

        elif kneeSide == 7 and imageNameWithoutLabel in imageNameList:
            side = 'right'
            predAngle = calculateAngle((hipPredX, hipPredY), (kneePredX, kneePredY), (anklePredX, anklePredY), side)
            index = imageNameList.index(imageNameWithoutLabel)
            rightAngleList[index] = predAngle

        # add text on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        # h, w, _ = imageArray.shape
        fontScale = 3
        thickness = 2
        index = imageNameList.index(imageNameWithoutLabel)
        left = leftAngleList[index]
        right = rightAngleList[index]

        # print(left, right)
        # print(imageArray)
        if (left == 'NA' and right != 'NA') or (left != 'NA' and right == 'NA'):
            imageArray = cv2.putText(imageArray, side+'Angle:' + str(predAngle), (50, 100),
                                     font, fontScale, greenColor, thickness)
        elif left != 'NA' and right != 'NA':
            imageArray = cv2.putText(imageArray, side+'Angle:' + str(predAngle), (50, 300),
                                     font, fontScale, greenColor, thickness)
        cv2.imwrite(visualizationPath + imageNameWithoutLabel, imageArray)
        if predAngle > 3 or predAngle <-3:
            cv2.imwrite(visualizationPath + imageNameWithoutLabel, imageArray)
        # print(kneeSide)
        # print( imageNameWithoutLabel)
        # print( imageNameList)
        # print(leftAngleList)
        # print(rightAngleList)
        # print(len(imageNameList), len(leftAngleList), len(rightAngleList))

        # if index == 5:
        #     break
    # print(len(imageNameList), len(leftAngleList), len(rightAngleList))
    # predInfo = {'imageNameWithoutLabel': imageNameList, 'leftAngle': leftAngleList, 'rightAngle': rightAngleList}
    # predDF = pd.DataFrame(predInfo)
    # predDF.to_csv(visualizationPath.replace('predictionVisualization/', '') + 'predInfo.csv', index=False)


def last(unetOutputCSVPath, roiCsvPath, annotationCsvPath, visualizationPath):
    hipDF = combineCSV(unetOutputCSVPath, roiCsvPath, annotationCsvPath, 'hip')
    kneeDF = combineCSV(unetOutputCSVPath, roiCsvPath, annotationCsvPath, 'knee')
    ankleDF = combineCSV(unetOutputCSVPath, roiCsvPath, annotationCsvPath, 'ankle')

    plotLandmark(hipDF, kneeDF, ankleDF, imageFolderPath, visualizationPath)


roiCsvPath = './roi.csv'
unetOutputCSVPath = '/infodev1/phi-data/shi/kneeX-ray/predictionBasedYoloOuput/unAnnotatedImage/experimentResult/'
annnotationCsvPath = '/infodev1/phi-data/shi/kneeX-ray/predictionBasedYoloOuput/unAnnotatedImage/data/'
imageFolderPath = '/infodev1/phi-data/shi/kneeX-ray/experiment202103/data/images/'
# visualizationPath = '/infodev1/phi-data/shi/kneeX-ray/predictionBasedYoloOuput/unAnnotatedImage/experimentResult/predictionVisualization/'
visualizationPath = '/infodev1/phi-data/shi/kneeX-ray/predictionBasedYoloOuput/unAnnotatedImage/experimentResult/big/'
last(unetOutputCSVPath, roiCsvPath, annnotationCsvPath, visualizationPath)

import pandas as pd
import cv2
import os
import math
import statistics

def predCoor(df):
    newPredXList, newPredYList, labelXList,labelYList = [],[],[],[]
    for index, row in df.iterrows():
        ratio = row['ratio']
        predX = row['predX']
        predY = row['predY']
        labelX =row['labelX']
        labelY = row['labelY']
        leftZero = row['leftZero']
        topZero = row['topZero']
        xStart = row['x']
        yStart = row['y']
        newPredX = xStart+ (int(predX*ratio)-leftZero)
        newPredY = yStart + (int(predY * ratio) - topZero)
        newLabelX = xStart + (int(labelX * ratio) - leftZero)
        newLabelY = yStart + (int(labelY * ratio) - topZero)
        newPredXList.append(newPredX)
        newPredYList.append(newPredY)
        labelXList.append(newLabelX)
        labelYList.append(newLabelY)
    df['newPredX']= newPredXList
    df['newPredY'] = newPredYList
    df['newLabelX'] = labelXList
    df['newLabelY'] = labelYList
    return df

def combineCSV(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, flag, sigma):
    unetOutputCSVPath = unetOutputCSVPath +str(sigma)+'/experimentResult/'
    rescaleInfoCSVPath = rescaleInfoCSVPath +str(sigma) + '/data/'

    yoloDF = pd.read_csv(yoloOutputCSVPath)
    unetDF = pd.read_csv(unetOutputCSVPath + flag + '/test/' + flag + 'PredInfo.csv')
    rescalDF = pd.read_csv(rescaleInfoCSVPath+flag  + '/newrescaleInfo.csv')
    tmpDF = pd.merge(yoloDF, unetDF, on='imageName')
    newDF = pd.merge(tmpDF,rescalDF, on='imageName')
    finalDF = predCoor(newDF)
    # finalDF.to_csv('./'+flag+'combo.csv', index=False)
    # print(finalDF.shape)
    return finalDF

def calculateAngle(hipCoor, kneeCoor, ankleCoor, leftOrRight):
    ang = math.degrees(
        math.atan2(ankleCoor[1] - kneeCoor[1], ankleCoor[0] - kneeCoor[0]) - math.atan2(hipCoor[1] - kneeCoor[1],
                                                                                        hipCoor[0] - kneeCoor[0]))
    # print(ang)
    if leftOrRight == 'left':
        return round(ang - 180,2)
    if leftOrRight == 'right':
        return -round((ang - 180),2)
def calculateDistance(x1,y1,x2,y2):
    return math.sqrt(pow((x1-x2),2)+pow((y1-y2),2))

def plotLandmark(hipDF, kneeDF, ankleDF, imageFolder, visualizationPath, sigma):

    visualizationPath2 = visualizationPath + str(sigma) + '/predictionVisualizationWithoutTxt/'
    if not os.path.exists(visualizationPath2):
        os.makedirs(visualizationPath2)

    visualizationPath = visualizationPath + str(sigma) + '/predictionVisualization/'
    if not os.path.exists(visualizationPath):
        os.makedirs(visualizationPath)

    imageNameList = []
    predLeftAngleList, predRightAngleList = [],[]
    labelLeftAngleList,labelRightAngleList = [],[]
    diffList = []
    hipDiffList, kneeDiffList, ankleDiffList = [],[],[]
    for index, row in hipDF.iterrows():
        imageName = row['imageNameListWithoutLabel']

        # print(index)
        # print(imageName)
        hipPredX, hipPredY,hipLabelX,hipLabelY = row['newPredX'],row['newPredY'],row['newLabelX'],row['newLabelY']
        kneePredX, kneePredY, kneeLabelX, kneeLabelY, kneeSide = kneeDF.iloc[index, 13], kneeDF.iloc[index, 14], kneeDF.iloc[index, 15],kneeDF.iloc[index, 16], kneeDF.iloc[index,2]
        anklePredX, anklePredY, ankleLabelX, ankleLabelY = ankleDF.iloc[index, 13], ankleDF.iloc[index, 14], ankleDF.iloc[index, 15],ankleDF.iloc[index, 16],
        # print(imageFolder+imageName, os.path.exists(imageFolder+imageName))
        hipDiffList.append(calculateDistance(hipLabelX,hipLabelY,hipPredX,hipPredY))
        kneeDiffList.append(calculateDistance(kneeLabelX, kneeLabelY, kneePredX, kneePredY))
        ankleDiffList.append(calculateDistance(ankleLabelX, ankleLabelY, anklePredX, anklePredY))

        imageArray = cv2.imread(imageFolder+imageName)
        pointRadius = 10
        redColor = (0, 0, 255)
        greenColor = (0, 255, 0)
        blueColor = (255, 0, 0)
        thickness = -1

        # mark center of label: blue
        cv2.circle(imageArray, (hipLabelX, hipLabelY), pointRadius, blueColor, thickness)
        cv2.circle(imageArray, (kneeLabelX, kneeLabelY), pointRadius, blueColor, thickness)
        cv2.circle(imageArray, (ankleLabelX, ankleLabelY), pointRadius, blueColor, thickness)
        #mark center of prediction, red
        cv2.circle(imageArray, (hipPredX, hipPredY), pointRadius, redColor, thickness)
        cv2.circle(imageArray, (kneePredX, kneePredY), pointRadius, redColor, thickness)
        cv2.circle(imageArray, (anklePredX, anklePredY), pointRadius, redColor, thickness)


        #draw line
        # draw red line among predicted points
        cv2.line(imageArray, (hipPredX,hipPredY), (kneePredX,kneePredY), redColor, 3)
        cv2.line(imageArray, (kneePredX,kneePredY), (anklePredX, anklePredY), redColor, 3)
        # draw blue line among predicted points
        cv2.line(imageArray, (hipLabelX, hipLabelY), (kneeLabelX, kneeLabelY), blueColor, 3)
        cv2.line(imageArray, (kneeLabelX, kneeLabelY), (ankleLabelX, ankleLabelY), blueColor, 3)

        cv2.imwrite(visualizationPath2 + imageName, imageArray)

        if kneeSide == 3 and imageName not in imageNameList:
            side = 'left'
            predAngle = calculateAngle((hipPredX,hipPredY), (kneePredX,kneePredY), (anklePredX, anklePredY), side)
            labelAngle = calculateAngle((hipLabelX,hipLabelY), (kneeLabelX,kneeLabelY), (ankleLabelX, ankleLabelY), side)
            predLeftAngleList.append(predAngle)
            predRightAngleList.append('NA')
            labelLeftAngleList.append(labelAngle)
            labelRightAngleList.append('NA')
            imageNameList.append(imageName)
        elif kneeSide == 3 and imageName in imageNameList:
            side = 'left'
            predAngle = calculateAngle((hipPredX, hipPredY), (kneePredX, kneePredY), (anklePredX, anklePredY), side)
            labelAngle = calculateAngle((hipLabelX, hipLabelY), (kneeLabelX, kneeLabelY), (ankleLabelX, ankleLabelY),
                                        side)
            index = imageNameList.find(imageName)

            predLeftAngleList[index] = predAngle
            labelLeftAngleList[index] = labelAngle

        elif kneeSide == 7 and imageName not in imageNameList :
            side = 'right'
            predAngle = calculateAngle((hipPredX, hipPredY), (kneePredX, kneePredY), (anklePredX, anklePredY), side)
            labelAngle = calculateAngle((hipLabelX, hipLabelY), (kneeLabelX, kneeLabelY), (ankleLabelX, ankleLabelY),side)
            predLeftAngleList.append('NA')
            predRightAngleList.append(predAngle)
            labelLeftAngleList.append('NA')
            labelRightAngleList.append(labelAngle)
            imageNameList.append(imageName)
        elif kneeSide == 7 and imageName not in imageNameList:
            side = 'right'
            predAngle = calculateAngle((hipPredX, hipPredY), (kneePredX, kneePredY), (anklePredX, anklePredY), side)
            labelAngle = calculateAngle((hipLabelX, hipLabelY), (kneeLabelX, kneeLabelY), (ankleLabelX, ankleLabelY),
                                        side)
            index = imageNameList.find(imageName)

            predRightAngleList[index] = predAngle
            labelRightAngleList[index] = labelAngle
        # add text on image

        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w, _ = imageArray.shape
        fontScale = 5
        thickness = 2

        imageArray = cv2.putText(imageArray, 'Angle diff:' + str(round(abs(predAngle-labelAngle),3)), (100, 200),
                               font, fontScale, greenColor, thickness)

        cv2.imwrite(visualizationPath + imageName, imageArray)


        diff = abs(predAngle -labelAngle)
        diffList.append(diff)

    distanceInfo = {'imageName': imageNameList, 'hipDiff': hipDiffList, 'kneeDiff': kneeDiffList,
                'ankleDiff': ankleDiffList}
    disDF = pd.DataFrame(distanceInfo)
    disDF.to_csv('***predictionBasedYoloOuput/512Sigma'+str(sigma)+'/postDisDiff.csv',index=False)
    disDF.to_csv('./postDisDiff.csv',
                 index=False)
    predInfo = {'imageName': imageNameList, 'predLeftAngle': predLeftAngleList, 'predAngleAngle': predRightAngleList,
                'labelLeftAngle': labelLeftAngleList,'labelRightAngle':labelRightAngleList}
    list(filter((2).__ne__, predRightAngleList+predLeftAngleList))
    print('pred angel mean and std: ', statistics.mean(list(filter(('NA').__ne__, predRightAngleList+predLeftAngleList))),
          statistics.stdev(list(filter(('NA').__ne__, predRightAngleList+predLeftAngleList))))
    print('label angel mean and std: ', statistics.mean(list(filter(('NA').__ne__, labelRightAngleList + labelLeftAngleList))),
          statistics.stdev(list(filter(('NA').__ne__, labelRightAngleList + labelLeftAngleList))))
    print('diff mean and std: ', statistics.mean(diffList), statistics.stdev(diffList))

    # predDF = pd.DataFrame(predInfo)
    # predDF.to_csv(visualizationPath.replace('predictionVisualization/','') + 'predInfo.csv', index=False)
    #
    # #print static
    # print('Sigma is ', sigma)
    # print('average angle difference is : ', sum(diffList)/len(diffList))
    count = len([i for i in diffList if i <= 1.5])
    print('number of angle difference smaller than 1.5 is: ', count, ' take ',
          round(count / len(diffList), 3), ' percentage \n')

def last(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, sigma):

    hipDF = combineCSV(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, 'hip', sigma)
    kneeDF = combineCSV(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, 'knee', sigma)
    ankleDF = combineCSV(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, 'ankle', sigma)
    plotLandmark(hipDF, kneeDF, ankleDF, imageFolderPath, visualizationPath, sigma)

yoloOutputCSVPath = './roi.csv'
unetOutputCSVPath = 'kneeX-ray/predictionBasedYoloOuput/512Sigma'
rescaleInfoCSVPath = 'kneeX-ray/predictionBasedYoloOuput/512Sigma'
imageFolderPath = 'kneeX-ray/experiment202103/data/images/'
visualizationPath = 'kneeX-ray/predictionBasedYoloOuput/512Sigma'


# last(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, sigma=10)
# last(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, sigma=15)
last(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, sigma=20)
# last(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, sigma=25)
# last(yoloOutputCSVPath, unetOutputCSVPath,rescaleInfoCSVPath, sigma=30)


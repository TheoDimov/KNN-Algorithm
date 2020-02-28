# Name: Todor Dimov
# Assignment: KNN Algorithm

from __future__ import division
import numpy as np
import pandas as pd

############################ Function Definitions ############################
# This function Z-score normalizes a given Training Data Frame.
def normalizeTrainDF(dataFrame):
    dfNormalized = dataFrame.copy()
    colList = list(dataFrame.columns)

    for col in range(len(colList)):
        colMean = dataFrame[colList[col]].mean()
        colStd = dataFrame[colList[col]].std()
        dfNormalized[colList[col]] = (dataFrame[colList[col]] - colMean)/colStd
    return dfNormalized

# This function for Z score normalization of the Test Data Frame using the Training Data Frame
def normalizeTestDF(testDataFrame, trainDataFrame):
    print(np.shape(testDataFrame))
    print(np.shape(trainDataFrame))
    dfNormalized = testDataFrame.copy()
    colList = list(testDataFrame.columns)

    for col in range(len(colList)):
        colMean = trainDataFrame[colList[col]].mean()
        colStd = trainDataFrame[colList[col]].std()
        dfNormalized[colList[col]] = (testDataFrame[colList[col]] - colMean)/colStd

    return dfNormalized

# This function provides the distance data frame
def getAllDistanceDF(trainDF , testDF):
    indexCounter = 0;
    appenedData = []

    for row in testDF.itertuples(index=False, name='Pandas'):
        distanceSeries = eculidDist(trainDF, row)
        testRowIndexList = [indexCounter]*np.shape(distanceSeries.values)[0]
        indexCounter += 1

        listToAddInFinalDF = { 'trainRowIndex' : np.array(distanceSeries.index) , 'distance': distanceSeries.values, 'testRowIndex' : testRowIndexList}
        dfEachTestRowDistance = pd.DataFrame(listToAddInFinalDF)
        appenedData.append(dfEachTestRowDistance)

    distanceDF = pd.concat(appenedData, axis=0)
    return distanceDF

# This function simply calculate the Euclidean distance of the training data frame and a test row
def eculidDist(trainDF , testRow):
    temp = (((trainDF.sub( testRow, axis=1))**2).sum(axis=1))**0.5
    temp.sort_values(axis=0, ascending=True, inplace=True)
    return temp
########################### Function Definitions End ############################

# adjustments to output display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# reading in dataframes
trainDF = pd.read_csv("spam_train.csv")
testDF = pd.read_csv("spam_test.csv")

# dropping the ID columns from DF
testDF.drop(testDF.columns[0] , axis=1, inplace=True)

# Creating a new DF for labels.
trainLable = trainDF[['class']].copy()
testLable = testDF[['Label']].copy()


# dropping the Label columns from both DF
trainDF.drop('class' , axis=1, inplace=True)
testDF.drop('Label' , axis=1, inplace=True)
distanceDF = getAllDistanceDF(trainDF,testDF)

kValuesList = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]

# Running a outer loop for all K Values
accuracyList = []
for kValue in range(len(kValuesList)) :
    indexCounter = 0
    predictedLabel = []
    for row in testDF.itertuples(index=False, name='Pandas'):
        distanceDFForRow = distanceDF[distanceDF['testRowIndex'] == indexCounter]
        nnIndex = distanceDFForRow.loc[:(kValuesList[kValue] - 1),'trainRowIndex']
        tmp = trainLable.iloc[nnIndex]['class'].value_counts()

        predictedLabel.append(tmp.idxmax())
        indexCounter += 1

    tmpList = {'Label' : predictedLabel}
    predictedTestLabel = pd.DataFrame(tmpList)

    differenceLabel = testLable.sub(predictedTestLabel , axis=1)
    accurateClassCount = len(differenceLabel[ differenceLabel['Label'] ==0 ])
    accuracyPercent = accurateClassCount/testLable['Label'].count()*100
    print('Accuracy for k=' ,kValuesList[kValue] , 'is:' , (accurateClassCount/testLable['Label'].count())*100, '%' )

    accuracyList.append(accuracyPercent)

tempDict = {'KValue': kValuesList, 'Accuracy %': accuracyList}
accuracyDF = pd.DataFrame(tempDict)

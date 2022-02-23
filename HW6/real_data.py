import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.svm import SVC
import pickle

def load_data():
	# load data 
	data = pd.read_csv(
		'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', header=None)
	p_ = data.shape[1]
	p = p_ - 1
	X = data.iloc[:, :p]
	y = data.iloc[:, p]
	return X.values, y.values

def getAccuracy(model,data,labels):
    predictions = model.predict(data)
    amountCorrect = sum(predictions==labels)
    accuracy = amountCorrect/len(labels)
    return accuracy

def tuneParamsXGB(estimators,maxdepths,trainData,trainLabels,testData,testLabels):
    trainAccuracies = []
    testAccuracies = []
    for estimator in estimators:
        for depth in maxdepths:
            model = xgb.XGBClassifier(n_estimator = estimator,max_depth = depth)
            model.fit(trainData, trainLabels)
            accTrain=getAccuracy(model,trainData,trainLabels)
            accTest=getAccuracy(model,testData,testLabels)
            trainAccuracies.append(accTrain)
            testAccuracies.append(accTest)
    return trainAccuracies,testAccuracies

def tuneParamsLinearSVC(CValues,maxIters,trainData,trainLabels,testData,testLabels):
    trainAccuracies = []
    testAccuracies = []
    for c in CValues:
        for iters in maxIters:
            model = SVC(kernel='linear',C = c, max_iter = iters)
            model.fit(trainData, trainLabels)
            accTrain=getAccuracy(model,trainData,trainLabels)
            accTest=getAccuracy(model,testData,testLabels)
            trainAccuracies.append(accTrain)
            testAccuracies.append(accTest)
    return trainAccuracies,testAccuracies
    
def tuneParamsGaussianSVC(CValues,maxIters,trainData,trainLabels,testData,testLabels):
    trainAccuracies = []
    testAccuracies = []
    for c in CValues:
        for iters in maxIters:
            model = SVC(kernel='rbf',C = c, max_iter = iters)
            model.fit(trainData, trainLabels)
            accTrain=getAccuracy(model,trainData,trainLabels)
            accTest=getAccuracy(model,testData,testLabels)
            trainAccuracies.append(accTrain)
            testAccuracies.append(accTest)
    return trainAccuracies,testAccuracies

def main():
    # Load data
    X, y = load_data()

    # TO DO:
    # Randomly split the data in to training set and testing test; 
    # Let testing set contain 20% of total dataset
    # You can check the train_test_split function in sklearn package
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # TO DO：
    # 1. Using the XgboostClassifier (default setting), report the training and testing accuracy
    # 2. Tuning the n_estimator and max_depth, compare the results
    xgbDict ={}
    xgbc = xgb.XGBClassifier()
    xgbc.fit(X_train, y_train)
    accTrain=getAccuracy(xgbc,X_train,y_train)
    accTest=getAccuracy(xgbc,X_test,y_test)
    xgbDict['defaultTrainAcc'] = accTrain
    xgbDict['defaultTestAcc'] = accTest
    
    n_estimators = [300,500,1000]
    maxDepths = [5,10,20]
    trainAccuracies,testAccuracies = tuneParamsXGB(n_estimators,maxDepths,X_train,y_train,X_test,y_test)
    xgbDict['trialTrainAcc'] = trainAccuracies
    xgbDict['trialTestAcc'] = testAccuracies
    pickle.dump(xgbDict, open( "xgbDict.p", "wb" ) )

    # TO DO：
    # 1. Using Linear SVM (default setting), report the training and testing accuracy
    # 2. Tuning C and max_iter, compare the results
    linearSVCDict ={}
    defaultSVC = SVC(kernel='linear',verbose=2)
    defaultSVC.fit(X_train, y_train)
    accTrain=getAccuracy(defaultSVC,X_train,y_train)
    accTest=getAccuracy(defaultSVC,X_test,y_test)
    linearSVCDict['defaultTrainAcc'] = accTrain
    linearSVCDict['defaultTestAcc'] = accTest
    
    CValues = [0.25,0.5,2]
    maxIters = [10000,100000,1000000]
    trainAccuracies,testAccuracies = tuneParamsLinearSVC(CValues,maxIters,X_train,y_train,X_test,y_test) 
    linearSVCDict['trialTrainAcc'] = trainAccuracies
    linearSVCDict['trialTestAcc'] = testAccuracies
    pickle.dump(linearSVCDict, open( "linearSVCDict.p", "wb" ) )    

    # TO DO：
    # 1. Using kernel SVM (with Gaussian Kernel) (default setting), report the training and testing accuracy
    # 2. Tuning C and max_iter, compare the results
    gaussianSVCDict ={}
    defaultGaussSVC = SVC(kernel='rbf',verbose=2)
    defaultGaussSVC.fit(X_train, y_train)
    accTrain=getAccuracy(defaultGaussSVC,X_train,y_train)
    accTest=getAccuracy(defaultGaussSVC,X_test,y_test)
    gaussianSVCDict['defaultTrainAcc'] = accTrain
    gaussianSVCDict['defaultTestAcc'] = accTest    
    
    CValues = [0.25,0.5,2]
    maxIters = [500,1000,10000]
    trainAccuracies,testAccuracies = tuneParamsGaussianSVC(CValues,maxIters,X_train,y_train,X_test,y_test)
    gaussianSVCDict['trialTrainAcc'] = trainAccuracies
    gaussianSVCDict['trialTestAcc'] = testAccuracies
    pickle.dump(gaussianSVCDict, open( "gaussianSVCDict.p", "wb" ) )


if __name__=="__main__":

 	# API usage 
 	main()

#%%
import pickle

xgbDict = pickle.load( open( "xgbDict.p", "rb" ) )
linearSVCDict = pickle.load( open( "linearSVCDict.p", "rb" ) )
gaussianSVCDict = pickle.load( open( "gaussianSVCDict.p", "rb" ) )
















import numpy as np

class SVM:
    def __init__(self):
        self.w = None
        self.b = None
        
    def predictSVM(self,X):
        output = np.dot(X,self.w)-self.b
        return np.sign(output)
    
    def trainSVM(self,X, Y,Xtest,Ytest,LAMB=0.0001, EPOCH=5000, LR=0.7):
    
         self.w = np.zeros((len(X[0]), 1))
         self.b = 0
         trainAccuracies =[]
         testAccuracies = []
         for e in range(EPOCH):
             for i in np.arange(X.shape[0]):
                 O = np.dot(X[i], self.w)-self.b
                 if (Y[i] * O < 1):
                     self.w = self.w - LR * (2 * LAMB * self.w - np.dot(X[i], Y[i]).reshape(-1,1))
                     self.b = self.b - LR * Y[i]
                 else:
                     self.w = self.w - LR * (2 * LAMB*self.w)
             predictionsTrain = self.predictSVM(X)
             accTrain = np.sum(predictionsTrain==Y.reshape(-1,1))/len(Y)
             trainAccuracies.append(accTrain)
             predictionsTest = self.predictSVM(Xtest)
             accTest = np.sum(predictionsTest==Ytest.reshape(-1,1))/len(Ytest)
             testAccuracies.append(accTest)
         return self.w,self.b,trainAccuracies,testAccuracies

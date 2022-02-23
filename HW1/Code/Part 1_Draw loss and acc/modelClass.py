import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool= nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = torch.flatten(out,1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
    def defineCriterionAndOptimizer(self,lr,mom):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=mom)
        self.criterion = criterion
        self.optimizer = optimizer
        
    def getLossAndAccuracyOnDataset(self,datasetLoader):
        totalcorrect = 0
        totalpred=0
        totalLoss = 0
        lossFcn = self.criterion
        # no gradients neededs
        with torch.no_grad():
            for batch in datasetLoader:
                images, labels = batch
                outputs = self(images.to(self.device))
                batchLoss = lossFcn(outputs, labels.to(self.device))
                totalLoss += batchLoss.item()*len(labels)
                _, predictions = torch.max(outputs.to(self.device), 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels.to(self.device), predictions.to(self.device)):
                    if label == prediction:
                        totalcorrect +=1
                    totalpred+=1
            avgDataLoss=totalLoss/totalpred
            dataAcc = (totalcorrect/totalpred)*100
            return avgDataLoss,dataAcc
                
        
    def trainNetwork(self,epochRange,trainLoader,testLoader):
        epochTrainLossProgress = []
        epochTrainAccProgress = []
        epochTestLossProgress = []
        epochTestAccProgress = []

        for epoch in range(epochRange):  

            running_loss = 0.0
            for i, data in enumerate(trainLoader, 0):
                inputs, labels = data
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs.to(self.device))
                loss = self.criterion(outputs, labels.to(self.device))
                loss.backward()
                self.optimizer.step()                
                # print statistics for each batch
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            epochTrainLoss,epochTrainAcc = self.getLossAndAccuracyOnDataset(trainLoader)
            epochTestLoss,epochTestAcc = self.getLossAndAccuracyOnDataset(testLoader)
            print("Train loss = %.3f Test loss = %.3f"%(epochTrainLoss,epochTestLoss))
            print("Train acc = %.3f percent Test acc = %.3f percent"%(epochTrainAcc,epochTestAcc))
            epochTrainLossProgress.append(epochTrainLoss)
            epochTrainAccProgress.append(epochTrainAcc)
            epochTestLossProgress.append(epochTestLoss)
            epochTestAccProgress.append(epochTestAcc)
            
        print('Finished Training')
        return epochTrainLossProgress,epochTrainAccProgress,epochTestLossProgress,epochTestAccProgress
    

        
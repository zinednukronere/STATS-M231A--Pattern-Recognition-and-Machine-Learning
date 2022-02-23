import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def testFunction(model,testLoader,classes):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    totalcorrect = 0
    totalpred=0
    totalLoss = 0
    lossFcn = nn.CrossEntropyLoss()
    # again no gradients neededs
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = model(images.to(model.device))
            batchloss = lossFcn(outputs, labels.to(model.device))
            totalLoss += batchloss.item()*len(images)
            _, predictions = torch.max(outputs.to(model.device), 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels.to(model.device), predictions.to(model.device)):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    totalcorrect +=1
                total_pred[classes[label]] += 1
                totalpred+=1


    # # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
    #                                                accuracy))

    accuracy = totalcorrect/totalpred
    print("Total accuracy = ",str(totalcorrect/totalpred))
    print("Loss = ",totalLoss/totalpred)
    return accuracy,correct_pred,total_pred,totalLoss/totalpred
    
    

class doubleChannelCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = None
        self.optimizer = None
        self.epochProgress=[]
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
        
    def trainNetwork(self,epochRange,trainLoader):
        epochLosses = []
        for epoch in range(epochRange):  
            epoch_loss = 0.0
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
                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (200)))
                    running_loss = 0.0
            print("Epoch loss is %.3f" % float(epoch_loss/(i+1)))
            epochLosses.append(float(epoch_loss/len(trainLoader)))
            self.epochProgress = epochLosses

        print('Finished Training')

class oneConvCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = None
        self.optimizer = None
        self.epochProgress=[]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    def forward(self,x):
        out = F.relu(self.conv1(x))        
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
        
    def trainNetwork(self,epochRange,trainLoader):
        epochLosses = []
        for epoch in range(epochRange):  
            epoch_loss = 0.0
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
                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (200)))
                    running_loss = 0.0
            print("Epoch loss is %.3f" % float(epoch_loss/(i+1)))
            epochLosses.append(float(epoch_loss/len(trainLoader)))
            self.epochProgress = epochLosses

        print('Finished Training')


class threeConvCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 26, 3)
        self.fc1 = nn.Linear(26 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = None
        self.optimizer = None
        self.epochProgress=[]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  
    
    def defineCriterionAndOptimizer(self,lr,mom):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=mom)
        self.criterion = criterion
        self.optimizer = optimizer
        
    def trainNetwork(self,epochRange,trainLoader):
        epochLosses = []
        for epoch in range(epochRange):  
            epoch_loss = 0.0
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
                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (200)))
                    running_loss = 0.0
            print("Epoch loss is %.3f" % float(epoch_loss/(i+1)))
            epochLosses.append(float(epoch_loss/len(trainLoader)))
            self.epochProgress = epochLosses

        print('Finished Training')
        
        
class leakyReluCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = None
        self.optimizer = None
        self.epochProgress=[]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    def forward(self,x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def defineCriterionAndOptimizer(self,lr,mom):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=mom)
        self.criterion = criterion
        self.optimizer = optimizer
        
    def trainNetwork(self,epochRange,trainLoader):
        epochLosses = []
        for epoch in range(epochRange):  
            epoch_loss = 0.0
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
                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (200)))
                    running_loss = 0.0
            print("Epoch loss is %.3f" % float(epoch_loss/(i+1)))
            epochLosses.append(float(epoch_loss/len(trainLoader)))
            self.epochProgress = epochLosses

        print('Finished Training')
        
class tanhCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = None
        self.optimizer = None
        self.epochProgress=[]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    def forward(self,x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def defineCriterionAndOptimizer(self,lr,mom):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=mom)
        self.criterion = criterion
        self.optimizer = optimizer
        
    def trainNetwork(self,epochRange,trainLoader):
        epochLosses = []
        for epoch in range(epochRange):  
            epoch_loss = 0.0
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
                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (200)))
                    running_loss = 0.0
            print("Epoch loss is %.3f" % float(epoch_loss/(i+1)))
            epochLosses.append(float(epoch_loss/len(trainLoader)))
            self.epochProgress = epochLosses

        print('Finished Training')
        
class originalCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = None
        self.optimizer = None
        self.epochProgress=[]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def defineCriterionAndOptimizer(self,lr,mom):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=mom)
        self.criterion = criterion
        self.optimizer = optimizer
        
    def trainNetwork(self,epochRange,trainLoader):
        epochLosses = []
        for epoch in range(epochRange):  
            epoch_loss = 0.0
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
                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (200)))
                    running_loss = 0.0
            print("Epoch loss is %.3f" % float(epoch_loss/(i+1)))
            epochLosses.append(float(epoch_loss/len(trainLoader)))
            self.epochProgress = epochLosses

        print('Finished Training')
    
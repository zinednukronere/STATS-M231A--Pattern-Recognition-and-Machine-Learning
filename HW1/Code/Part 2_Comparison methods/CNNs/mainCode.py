#%% Importing the libraries
import torch
import torchvision
import torchvision.transforms as transforms
#%% Lading test and train data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataDirectory = r'C:\Users\orkun\Desktop\UCLA Courses\STATS M231A Pattern Recog & ML\HW1\Code\data'

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root=dataDirectory, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root=dataDirectory, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%% Loading the different cnn classes
from cnnDefiner import *
#%% Train nn with double channels and get performance
net = doubleChannelCNN()    
net.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './netDoubleChannel.pth'
torch.save(net.state_dict(),PATH)
accuracyTR,correct_predTR,total_predTR,lossTR = testFunction(net,trainloader,classes)
accuracy,correct_pred,total_pred,loss = testFunction(net,testloader,classes)
#%% Train nn with 1 conv layer and get performance
net = oneConvCNN()    
net.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './netOneConv.pth'
torch.save(net.state_dict(),PATH)
accuracyTR,correct_predTR,total_predTR,lossTR = testFunction(net,trainloader,classes)
accuracy,correct_pred,total_pred,loss = testFunction(net,testloader,classes)
#%% Train nn with 3 conv layers and get performance
net = threeConvCNN()    
net.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './netThreeConv.pth'
torch.save(net.state_dict(),PATH)
accuracyTR,correct_predTR,total_predTR,lossTR = testFunction(net,trainloader,classes)
accuracy,correct_pred,total_pred,loss = testFunction(net,testloader,classes)
#%% Train leaky relu cnn and get performance
net = leakyReluCNN()    
net.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './leakyRealuConv.pth'
torch.save(net.state_dict(),PATH)
accuracyTR,correct_predTR,total_predTR,lossTR = testFunction(net,trainloader,classes)
accuracy,correct_pred,total_pred,loss = testFunction(net,testloader,classes)
#%% Train tanh cnn and get performance
net = tanhCNN()    
net.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './tanhConv.pth'
torch.save(net.state_dict(),PATH)
accuracyTR,correct_predTR,total_predTR,lossTR = testFunction(net,trainloader,classes)
accuracy,correct_pred,total_pred,loss = testFunction(net,testloader,classes)

#%% Train original cnn with less learning rate and get performance
net = originalCNN()    
net.defineCriterionAndOptimizer(lr=0.0001, mom=0.9)
net.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './lessLearnRateConv.pth'
torch.save(net.state_dict(),PATH)
accuracyTR,correct_predTR,total_predTR,lossTR = testFunction(net,trainloader,classes)
accuracy,correct_pred,total_pred,loss = testFunction(net,testloader,classes)

#%% Train original cnn with more learning rate and get performance
net = originalCNN()    
net.defineCriterionAndOptimizer(lr=0.01, mom=0.9)
net.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './moreLearnRateConv.pth'
torch.save(net.state_dict(),PATH)
accuracyTR,correct_predTR,total_predTR,lossTR = testFunction(net,trainloader,classes)
accuracy,correct_pred,total_pred,loss = testFunction(net,testloader,classes)





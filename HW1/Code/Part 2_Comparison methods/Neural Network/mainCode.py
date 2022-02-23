#%% Load the libraries
import torch
import torchvision
import torchvision.transforms as transforms
#%% Load the test and train data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 32

dataDirectory = r'C:\Users\orkun\Desktop\UCLA Courses\STATS M231A Pattern Recog & ML\HW1\Code\data'

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
#%% Load each different FCNN model
from neuralNetworkDefiner import Net2Layer,Net3Layer,Net4Layer,Net5Layer,testFunction
#%% Train nn with 2 layers
net2 = Net2Layer()    
net2.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net2.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './2layers.pth'
torch.save(net2.state_dict(),PATH)
#%% Train nn with 3 layers
net3 = Net3Layer()    
net3.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net3.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './3layers.pth'
torch.save(net3.state_dict(),PATH)
#%% Train nn with 4 layers
net4 = Net4Layer()    
net4.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net4.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './4layers.pth'
torch.save(net4.state_dict(),PATH)
#%% Train nn with 5 layers
net5 = Net5Layer()    
net5.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net5.trainNetwork(epochRange=10,trainLoader=trainloader)
PATH = './5layers.pth'
torch.save(net5.state_dict(),PATH)

#%% Load a network at a path and use that network to get performance on test and
#train sets
PATH = './5layers.pth'
net = Net5Layer()
net.load_state_dict(torch.load(PATH))
accuracyTR,correct_predTR,total_predTR,testLossTR = testFunction(net,trainloader,classes)
accuracy,correct_pred,total_pred,testLoss = testFunction(net,testloader,classes)









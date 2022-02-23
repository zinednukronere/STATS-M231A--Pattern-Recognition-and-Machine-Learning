#%%Import the necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#%% Load the image dataset for training and testing
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

#%% Train the model
from modelClass import Net
net = Net()    
net.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
[epochTrainLossProgress,epochTrainAccProgress,
 epochTestLossProgress,epochTestAccProgress]=net.trainNetwork(10,trainloader,testloader)
#%% Save model and progress
import pickle
progressInfo = [epochTrainLossProgress,epochTrainAccProgress,
 epochTestLossProgress,epochTestAccProgress]
pickle.dump(net, open( "net.p", "wb" ) )
pickle.dump(progressInfo, open("progressInfo.p", "wb" ) )

#%% Load 
net = pickle.load(open("net.p","rb"))
progressInfo = pickle.load(open("progressInfo.p","rb"))
#%% Plot loss vs epoch graph
epochTrainLossProgress = progressInfo[0]
epochTestLossProgress = progressInfo[2]

plt.plot(epochTrainLossProgress,"b")
plt.plot(epochTestLossProgress,"g")
plt.legend(["Training loss", "Testing loss"])
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.title("Change in train & test loss through epochs")

#%% Plot accuacy epoch graph
epochTrainAccProgress = progressInfo[1]
epochTestAccProgress = progressInfo[3]

plt.plot(epochTrainAccProgress,"b")
plt.plot(epochTestAccProgress,"g")
plt.legend(["Training accuracy", "Testing accuracy"])
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy (%)")
plt.title("Change in train & test accuracy through epochs")

#%% Get two random images from each class in the test set. Try the model on
# these images
import numpy as np
import torchvision.transforms.functional as F
from random import sample
import random

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
random.seed(10)
testLabels = testset.targets
testImages = testset.data
randomPlaneData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 0],2),:,:,:])
randomCarData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 1],2),:,:,:])
randomBirdData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 2],2),:,:,:])
randomCatData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 3],2),:,:,:])
randomDeerData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 4],2),:,:,:])
randomDogData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 5],2),:,:,:])
randomFrogData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 6],2),:,:,:])
randomHorseData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 7],2),:,:,:])
randomShipData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 8],2),:,:,:])
randomTruckData = np.array(testImages[sample([i for i, x in enumerate(testLabels) if x == 9],2),:,:,:])
randomData = np.concatenate([randomPlaneData,randomCarData,randomBirdData,randomCatData,
                             randomDeerData,randomDogData,randomFrogData,randomHorseData,
                             randomShipData,randomTruckData])
a = torch.empty((1,3,32,32))
for k in range(0,randomData.shape[0]):
    pil_image = F.to_pil_image(randomData[k,:,:,:])
    image_tensor = transform(pil_image).float()
    image_tensor = image_tensor.unsqueeze(0)
    a=torch.cat((a,image_tensor),0)
    
imshow(torchvision.utils.make_grid(a[1:,:,:,:],5))
outputs = net(a[1:].to(net.device))
_, predicted = torch.max(outputs.data, 1)
print(predicted)
#%% Get first 20 images from the first minibatch of the test set 
#(This secion not used in report. But it is what most people will do for the HW)
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images[0:20],5))
outputs = net(images[0:20].to(net.device))
_, predicted = torch.max(outputs.data, 1)
print(predicted)
print(labels[0:20])


#%% Try the model on the test data
correct = 0
total = 0
softmax = torch.nn.Softmax(dim=1)
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images.to(net.device))
        outputProbs = softmax(outputs.to(net.device))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputProbs.data, 1)
        total += labels.size(0)
        correct += (predicted.to(net.device) == labels.to(net.device)).sum().item()

print('Accuracy of the network on the 10000 test images: %.3f %%' % (
    100 * correct / total))


    
    
    
    
    
    
    
    
    
    
    
    
    
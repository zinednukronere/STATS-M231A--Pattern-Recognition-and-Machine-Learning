#%% 
import torch
import torchvision
import torchvision.transforms as transforms
#%%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next() 

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
#%%
from neuralNetworkClass import Net
net = Net()    
net.defineCriterionAndOptimizer(lr=0.001, mom=0.9)
net.trainNetwork(2,trainloader)
#%%
PATH = './2layers.pth'
torch.save(net.state_dict(),PATH)
#%%
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%%
net = Net()
net.load_state_dict(torch.load(PATH))
#%%
correct = 0
total = 0
softmax = torch.nn.Softmax(dim=1)
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = prevNet(images.to(net.device))
        outputProbs = softmax(outputs.to(net.device))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputProbs.data, 1)
        total += labels.size(0)
        correct += (predicted.to(net.device) == labels.to(net.device)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#%%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
totalcorrect = 0
totalpred=0
# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(net.device))
        _, predictions = torch.max(outputs.to(net.device), 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels.to(net.device), predictions.to(net.device)):
            if label == prediction:
                correct_pred[classes[label]] += 1
                totalcorrect +=1
            total_pred[classes[label]] += 1
            totalpred+=1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))

print("Total accuracy = ",str(totalcorrect/totalpred))
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    
    
    
    
    
    
    
    
    
    
    
    
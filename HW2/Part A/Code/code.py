#%%
from PyTorch_CIFAR10_master.cifar10_models import resnet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import pickle

#%%
resnet18 = resnet.resnet18(pretrained=True)
resnet34 = resnet.resnet34(pretrained=True)
resnet50 = resnet.resnet50(pretrained=True)
#%%
meanCIFAR = [0.4914, 0.4822, 0.4465]
stdCIFAR = [0.2471, 0.2435, 0.2616]
normalize = transforms.Normalize(mean=meanCIFAR,
                                 std=stdCIFAR)
transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize])
batch_size = 32
dataDirectory = r'C:\Users\orkun\Desktop\UCLA Courses\STATS M231A Pattern Recog and ML\HW2\Data'
testset = torchvision.datasets.CIFAR10(root=dataDirectory, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

#%% 
def testFunction(model,testLoader):
    totalcorrect = 0
    totalpred=0
    totalLoss = 0
    lossFcn = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # again no gradients neededs
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = model(images)
            batchloss = lossFcn(outputs, labels)
            totalLoss += batchloss.item()*len(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    totalcorrect +=1
                totalpred+=1
    accuracy = totalcorrect/totalpred
    print("Total accuracy = ",str(totalcorrect/totalpred))
    print("Loss = ",totalLoss/totalpred)
    return accuracy,totalLoss/totalpred
#%%
resnet18.eval()
resnet34.eval()
resnet50.eval()
res18Acc,res18Loss = testFunction(resnet18,testloader)
res34Acc,res34Loss = testFunction(resnet34,testloader)
res50Acc,res50Loss = testFunction(resnet50,testloader)
accuracies = [res18Acc,res34Acc,res50Acc]
losses = [res18Loss,res34Loss,res50Loss]
pickle.dump(accuracies, open( "accuracies.p", "wb" ) )
pickle.dump(losses, open( "losses.p", "wb" ) )

#%%
accuracies = pickle.load(open("accuracies.p","rb"))
models = ['ResNet18', 'ResNet34', 'ResNet50']
ax=plt.bar(models,[x * 100 for x in accuracies])
i = 1.0
j = 1
for i in range(len(accuracies)):
    plt.annotate(str(accuracies[i]*100) + "%", (models[i], accuracies[i]*100+j))
plt.xlabel("Model Type")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracies on CIFAR-10 Test Data")
plt.show()
#%%
losses = pickle.load(open("losses.p","rb"))
models = ['Resnet18', 'Resnet34', 'Resnet50']
plt.bar(models,losses)
plt.xlabel("Model Type")
plt.ylabel("Loss")
plt.title("Model Loss on CIFAR-10 Test Data")
plt.show()

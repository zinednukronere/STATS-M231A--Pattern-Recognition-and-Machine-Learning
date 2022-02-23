import numpy as np
from matplotlib import pyplot as plt
from svmImplementation import SVM
from sklearn.model_selection import train_test_split

# Generate data
def load_data():
	# generate random 2-D data
	X_pos = np.random.randn(1000, 2) * np.array([[0.7, 0.8]]) + np.array([[-2.0, -2.0]])
	X_neg = np.random.randn(1000, 2) * np.array([[1.0, 0.8]]) + np.array([[1.25, 1.25]])
	X = np.concatenate([X_pos, X_neg], axis=0)
	y = np.concatenate([np.ones(1000), -np.ones(1000)])
	return X, y

# here is the visualization of data
X, y = load_data()
plt.figure()
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='class 1')
plt.scatter(X[y==-1][:, 0], X[y==-1][:, 1], c='blue', label='class 0')
plt.legend()
plt.savefig('toy.png')

# TO DO:
# Randomly split the data in to training set and testing test; 
# Let testing set contain 20% of total dataset
# You can check the train_test_split function in sklearn package
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# TO DO:
# Implement linear SVM from scratch;
# Plot the curve of training accuracy and testing accuracy
svm = SVM()
w,b,trainAccuracies,testAccuracies = svm.trainSVM(X_train,y_train,
                                                  X_test,y_test,
                                                  LAMB=0.001, EPOCH=10, LR=0.001)

#%%
plt.plot([element * 100 for element in trainAccuracies],label = "Training accuracy")
plt.plot([element * 100 for element in testAccuracies],label = "Testing accuracy")
plt.ylabel('Accuracy (%)')
plt.xlabel('Training iterations')
plt.title("Change of training and testing accuracy through training iterations")
plt.legend()
plt.show()

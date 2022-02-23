#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import plot_confusion_matrix,confusion_matrix
import seaborn as sns

#%%
df = pd.read_csv("iris.csv")
#%%
oneHot = pd.get_dummies(df["Name"])
features = df.iloc[:,0:4]
#%% Shuffle then train test split
XTr,XTest,YTr,YTest = train_test_split(features,
                                       oneHot,test_size=0.2,random_state=42)
#%% Fit model on train data
clf = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(gamma='auto')))
scores = cross_val_score(clf, XTr, YTr, cv=5)
#%% Cross validation completed. Fitting model
clf.fit(XTr,YTr)
#%% Test model on test data
predictions = np.array(clf.predict(XTest)).argmax(axis=1)
actualResults = np.array(YTest).argmax(axis=1)
cm = confusion_matrix(actualResults, predictions)
f = sns.heatmap(cm, annot=True, fmt='d')

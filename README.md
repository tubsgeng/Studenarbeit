# Studenarbeit
Experimental investigation of product and process parameters in the lamination process of separators and electrodes using a model-based approach 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 8:00:12 2022

@author: gengliu
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('/Users/gengliu/Desktop/003.csv',index_col=0,sep = ";")
data.describe()
X = data.iloc[:,data.columns !="Voltage"]
y = data.iloc[:,data.columns =="Voltage"]
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=6)

#Decision Tree , Random Forest
for i in [Xtrain,Xtest,ytrain,ytest]:
    i.index = range(i.shape[0])
clf = DecisionTreeClassifier(random_state = 25,class_weight = 'balanced')
clf = clf.fit(Xtrain,ytrain)
score = clf.score(Xtest,ytest)
score
y_pred=clf.predict(Xtest)
clf.feature_importances_
confmat = confusion_matrix(y_true=ytest, y_pred=y_pred)  # 输出混淆矩阵
print(confmat)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.title('Confusion matrix for our classifier')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

print('precision:%.3f' % precision_score(y_true=ytest, y_pred=y_pred,average= 'macro'))
print('recall:%.3f' % recall_score(y_true=ytest, y_pred=y_pred,average= 'macro'))
print('F1:%.3f' % f1_score(y_true=ytest, y_pred=y_pred,average= 'macro'))
from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(clf, Xtest, ytest,
                               cmap=plt.cm.Blues,
                               normalize='true')
plt.title('Confusion matrix for our classifier')


plt.figure(figsize=(8, 10))
plt.savefig("/Users/gengliu/Desktop/bar.png")
plt.show(matrix)

# Evaluate predictions
print(accuracy_score(ytest,y_pred))
print(confusion_matrix(ytest,y_pred))
print(classification_report(ytest,y_pred))

from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(random_state = 25,class_weight = 'balanced')
score = cross_val_score(clf,X,y,cv=10).mean()
score
tr=[]
te=[]
for i in range(10):
    clf = DecisionTreeClassifier(random_state = 25
                                ,max_depth = i+1
                                ,criterion='entropy'
                                ,class_weight = 'balanced')
    clf = clf.fit(Xtrain,ytrain)
    score_tr = clf.score(Xtrain,ytrain)
    score_te = cross_val_score(clf,X,y,cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
plt.plot(range(1,11),tr,color='red',label='train')
plt.plot(range(1,11),te,color='blue',label='test')
plt.xticks(range(1,11))
plt.legend()
plt.show()

# matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
clf = DecisionTreeClassifier(random_state=0,class_weight = 'balanced')
rfc = RandomForestClassifier(random_state=0,class_weight = 'balanced')
clf = clf.fit(Xtrain,ytrain)
rfc = rfc.fit(Xtrain,ytrain)
score_c=clf.score(Xtest,ytest)
score_r=rfc.score(Xtest,ytest)
print('Single Tree:{}'.format(score_c)
     ,'Random Forest:{}'.format(score_r))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(Xtrain, ytrain)
y_pre = knn_model.predict(Xtest)

print("a", ytest)
print("ergebnisse", y_pre)
conf_mat = confusion_matrix(ytest, y_pre)
print(conf_mat)
print(classification_report(ytest, y_pre))
# Generate confusion matrix
from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(knn_model, Xtest, ytest,
                               cmap=plt.cm.Blues,
                               normalize='true')
plt.title('Confusion matrix for our classifier')


plt.figure(figsize=(8, 10))

plt.show(matrix)

#SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
clf = SVC(C=1,kernel='rbf',gamma=1)

clf.fit(Xtrain, ytrain.values.ravel())
print (clf.score(Xtrain, ytrain.values.ravel())) 
print ('Acc_Train', accuracy_score(ytrain.values.ravel(), clf.predict(Xtrain)))
print (clf.score(Xtest, ytest))
print ('Acc_Test：', accuracy_score(ytest.values.ravel(), clf.predict(Xtest)))
y_pred=clf.predict(Xtest)
print(y_pred)
# Evaluate predictions
print(accuracy_score(ytest,y_pred))
print(confusion_matrix(ytest,y_pred))
print(classification_report(ytest,y_pred))
# Generate confusion matrix
from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(clf, Xtest, ytest,
                               cmap=plt.cm.Blues,
                               normalize='true')
plt.title('Confusion matrix for our classifier')


plt.figure(figsize=(8, 10))
plt.savefig("/Users/gengliu/Desktop/bar.png")
plt.show(matrix)


#ROC AUC
from sklearn import metrics
data = pd.read_csv('/Users/gengliu/Desktop/003.csv',index_col=0,sep = ";")

X = data.iloc[:,data.columns !="Voltage"]
y = data.iloc[:,data.columns =="Voltage"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6)
#set up plotting area


plt.figure(figsize=(6,4),dpi=150)
plt.grid()

#plt.figure(0).clf()

#fit Decision and plot ROC curve
model = DecisionTreeClassifier(random_state = 25,class_weight = 'balanced')
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="decision tree, AUC="+str(auc))

#KNN
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="KNN, AUC="+str(auc))

#SVC
model = SVC(C=1,kernel='rbf',gamma=1,probability=True)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="SVM, AUC="+str(auc))
plt.legend()

#RD
model = RandomForestClassifier(random_state = 25)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)


plt.plot(fpr,tpr,label="random forest, AUC="+str(auc))

plt.xlabel('False positive Rate')
plt.ylabel('True positive Rate')
plt.legend()

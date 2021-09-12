"""
Emotion Classifier - SVM
MFCC-SVM model with json data

Created on Sun Apr 19 12:41:09 2020
Author: Qiyang Ma

Refactored by: Christopher Woloshyn
"""

import json
import random as rd
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC


def load_data(filename):
    """Load MFCC data from json file (No Noise, Light Noise, Heavy Noise)"""
    f = open(filename)
    dic = json.load(f)
    f.close()

    data = dic['mfcc']
    labels = dic['labels']
    X, y = [], []
    l = list(range(len(data)))
    rd.shuffle(l)

    for i in l:
        ds = data[i]
        tmp = []
        for line in ds:
            tmp.extend(line)
        X.append(tmp)
        if labels[i] in [7, 8, 9, 10, 11, 12, 13]:
            labels[i] -= 7
        y.append(labels[i])
    X = np.array(X)
    y = np.array(y)

    return X, y

def instance_svm(X, y):
    """Creates an SVM from the preprocessed data and returns the RoC params."""
    kf = KFold(n_splits=5)   
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    c = list(range(7))

    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]   
        clf = SVC(C=50, kernel='rbf', gamma='scale')
        yscore = clf.fit(Xtrain, ytrain).decision_function(Xtest)
        ytest = label_binarize(ytest, classes=c)
        fpr, tpr, _ = roc_curve(ytest.ravel(), yscore.ravel())
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    return mean_fpr, mean_tpr, mean_auc

def calc_accuracy(X, y):
    """Calculates the mean accuracy of the model."""
    kf = KFold(n_splits=5)
    res = [] 
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]   
        clf = SVC(C=50, kernel='rbf', gamma='scale')
        output = clf.fit(Xtrain, ytrain).predict(Xtest)
        n, Num = 0, len(ytest)

        for i in range(Num):
            if output[i] == ytest[i]:
                n += 1
                accuracy = n / Num
        
        res.append(accuracy)
    mean_acc = np.mean(res)

    return mean_acc
    
def roc(fpr, tpr, auc, color='darkorange', text=''):
    """Plots the RoC curve for a particular model."""
    lw = 2
    plt.plot( fpr, tpr, color=color, lw=lw, label=f'{text} (area = {auc})')

def main():
    """Instance an SVM classifier for each noise level and plot performance."""
    path = f'{sys.path[0]}/../data/json/'
    # No Noise.
    Xn, yn = load_data(f'{path}no_noise.json')
    fprn, tprn, aucn = instance_svm(Xn, yn)
    accn = calc_accuracy(Xn, yn)
    # Light Noise.
    Xl, yl = load_data(f'{path}heavy_noise.json')
    fprl, tprl, aucl = instance_svm(Xl, yl)
    accl = calc_accuracy(Xl, yl)
    # Heavy Noise.
    Xh, yh = load_data(f'{path}light_noise.json')
    fprh, tprh, auch = instance_svm(Xh, yh)
    acch = calc_accuracy(Xh, yh)

    # Plot Results.
    plt.figure()
    roc(fprn, tprn, aucn, color='darkgreen', text='No noise')
    roc(fprh, tprh, auch, color='yellow', text='Heavy noise')
    roc(fprl, tprl, aucl, color='IndianRed', text='Light noise')    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of Multi-classes from MFCC')
    plt.legend(loc="lower right")
    plt.show()

    return accn, acch, accl

if __name__ == '__main__':
    main()

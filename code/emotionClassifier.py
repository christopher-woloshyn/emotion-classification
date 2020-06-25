# -*- coding: utf-8 -*-
"""
emotion Classifier - svm
MFCC-SVM model with json data

Functions:
loadData - load MFCC data from json file ( No noisy, Light noisy, Heavy noisy )
SVMclassifier - classify emotions by using SVM with 5-fold cross-validation
Acc - calculate the accuracy from MFCC-SVM model
roc - plot ROC curve
main - main function

Created on Sun Apr 19 12:41:09 2020
Author: Qiyang Ma
"""

#import pandas as pd
import numpy as np
import random as rd
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
import json

def loadData( filename ):
    # json file
    f = open( filename )
    dic = json.load( f )
    data = dic[ 'mfcc' ]
    labels = dic[ 'labels' ]
    X, y = [], []
    l = list( range( len( data ) ) )
    rd.shuffle( l )
    for i in l:
        ds = data[ i ]
        tmp = []
        for line in ds:
            tmp.extend( line )
        X.append( tmp )
        if labels[ i ] in [ 7, 8, 9, 10, 11, 12, 13 ]:
            labels[ i ] -= 7
        y.append( labels[i] )
    X = np.array( X )
    y = np.array( y )
    return X, y

def SVMclassifier( X, y ):
    # Inputs:
    # X - MFCC feature matrix
    # y - Labels as different emotion
    # Outputs:
    # mean_fpr, mean_tpr, mean_auc - the parameter for drawing roc
    kf = KFold( n_splits = 5 )   
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    c = list( range( 7 ) )
    for trainIndex, testIndex in kf.split( X ):
        Xtrain, Xtest = X[ trainIndex ], X[ testIndex ]
        ytrain, ytest = y[ trainIndex ], y[ testIndex ]   
        clf = SVC( C = 50, kernel = 'rbf', gamma = 'scale' )
        yscore = clf.fit( Xtrain, ytrain ).decision_function( Xtest )
        ytest = label_binarize( ytest, classes = c )
        fpr, tpr, _ = roc_curve( ytest.ravel(), yscore.ravel() )
        interp_tpr = interp( mean_fpr, fpr, tpr )
        interp_tpr[0] = 0.0
        tprs.append( interp_tpr )
    mean_tpr = np.mean( tprs, axis = 0 )
    mean_tpr[ -1 ] = 1.0
    mean_auc = auc( mean_fpr, mean_tpr )
    return mean_fpr, mean_tpr, mean_auc

def Acc( X, y ):
    # Outputs;
    # mean_acc - accuracy of models
    kf = KFold( n_splits = 5 )
    res = [] 
    for trainIndex, testIndex in kf.split( X ):
        Xtrain, Xtest = X[ trainIndex ], X[ testIndex ]
        ytrain, ytest = y[ trainIndex ], y[ testIndex ]   
        clf = SVC( C = 50, kernel = 'rbf', gamma = 'scale' )
        output = clf.fit( Xtrain, ytrain ).predict( Xtest )
        n, Num = 0, len( ytest )
        for i in range( Num ):
            if output[i] == ytest[ i ]:
                n += 1
                accuracy = n/Num
        res.append( accuracy )
    mean_acc = np.mean( res )
    return mean_acc
    
def roc( fpr, tpr, auc, color = 'darkorange', text = '' ):
    lw = 2
    plt.plot( fpr, tpr, color = color, lw = lw, \
             label = text + ' (area = %0.2f)'% auc )

def main():
    Xn, yn = loadData( 'data-no-noise.json' )
    fprn, tprn, aucn = SVMclassifier( Xn, yn )
    accn = Acc( Xn, yn )
    Xh, yh = loadData( 'data-heavy-noise.json' )
    fprh, tprh, auch = SVMclassifier( Xh, yh )
    acch = Acc( Xh, yh )
    Xl, yl = loadData( 'data-light-noise.json' )
    fprl, tprl, aucl = SVMclassifier( Xl, yl )
    accl = Acc( Xl, yl )
    plt.figure()
    roc( fprn, tprn, aucn, color = 'darkgreen', text = 'No noise' )
    roc( fprh, tprh, auch, color = 'yellow', text = 'heavy noise' )
    roc( fprl, tprl, aucl, color = 'IndianRed', text = 'light noise' )    
    plt.plot( [ 0, 1 ], [ 0, 1 ], color = 'navy', lw = 2, linestyle = '--' )
    plt.xlim( [ 0.0, 1.0 ] )
    plt.ylim( [ 0.0, 1.05 ] )
    plt.xlabel( 'False Positive Rate' )
    plt.ylabel( 'True Positive Rate' )
    plt.title( 'ROC of Multi-classes from MFCC' )
    plt.legend( loc = "lower right" )
    plt.show()
    return accn, acch, accl



















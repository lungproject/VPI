from __future__ import print_function

import logging
import os

import pandas as pd

import numpy as np
import math
from scipy import stats
import pandas
# import tensorflow as tf
# import keras.backend as K
from sklearn import metrics
import matplotlib.pyplot as plt
import copy



def combinepatch(y_pred):
    mu0, std0 = stats.norm.fit(y_pred)
    inp1 = y_pred[np.where(y_pred>=0.4972)]
    inp2 = y_pred[np.where(y_pred<0.4972)]
    if np.shape(y_pred)[0]>2:
        if np.shape(inp1)[0] > np.shape(inp2)[0]:
            ind1 = np.where((inp1>mu0-std0)&(inp1<mu0+std0))
            pre = np.mean(inp1[ind1])
        else:
            ind2 = np.where((inp2>mu0-std0)&(inp2<mu0+std0))
            pre = np.mean(inp2[ind2])
    else:
        pre = np.mean(y_pred)

    return pre



if __name__ == '__main__':
    RS = np.load('./results/predictRS.npy')
    outPath = r''

    inputCSV = os.path.join(outPath, './data/testCases.csv')
    flists = pandas.read_csv(inputCSV).T


    inputCSV = os.path.join(outPath, './data/Clinic.csv')
    Clinics = pandas.read_csv(inputCSV).T
    IDs=[];
    diameters=[];noduletypes=[];IDclinics=[];Labels=[];
    for entry in flists:  # Loop over all columns (i.e. the test cases)
        IDs = np.append(IDs,flists[entry]['ID'])


    for entry in Clinics:  # Loop over all columns (i.e. the test cases)
        IDclinics = np.append(IDclinics,Clinics[entry]['ID'])
        diameters = np.append(diameters,Clinics[entry]['diameter'])
        noduletypes = np.append(noduletypes, Clinics[entry]['noduletype'])
        Labels = np.append(Labels, Clinics[entry]['Label'])

    ID = set(IDs)
    allRS = [];
    ytrue = [];
    for entry in ID:
        print(entry)
        ind = np.where(IDs==entry)
        nRS=combinepatch(RS[ind])
        inc = np.where(IDclinics==entry)
        nRS2 = 1/(1+math.exp(-8*nRS+0.166*round(diameters[inc][0])-0.909*noduletypes[inc][0]+5.925))
        print(nRS2)
        allRS = np.append(allRS,nRS2)
        ytrue = np.append(ytrue,Labels[inc][0])


    score = copy.deepcopy(allRS)
    score[allRS>=0.5433] = 1
    score[allRS<0.5433] = 0
    print(score)

    fpr, tpr, thresholds = metrics.roc_curve(ytrue,  allRS, pos_label=1)
    accuracy = metrics.accuracy_score(ytrue, score)
    auc = metrics.auc(fpr, tpr)
    mcm = metrics.multilabel_confusion_matrix(ytrue, score)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    sen = tp/(tp+fn)
    sen = sen[1]
    spec = fp/(fp+tn)
    spec = 1-spec[1]

    f1_score = metrics.f1_score(ytrue, score, average='macro')
    recall_score = metrics.recall_score(ytrue, score, average='macro')

    print("auc:"+str(auc))
    print("sensitivity:" + str(sen))
    print("specificity:" + str(spec))
    print("accuracy" + str(accuracy))

    # Plot the AUC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

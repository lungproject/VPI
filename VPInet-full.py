#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import shutil
import scipy.misc
from scipy.ndimage import zoom
import nrrd
import logging
import torch
import pandas
import tensorflow as tf
from keras import backend as K

from keras.models import load_model
from dropblockmw import DropBlock2Dmw


import math
from scipy import stats

from sklearn import metrics
import copy
import warnings

warnings.filterwarnings('ignore')


def _boundingBox(A):
    B = np.argwhere(A)
    if A.ndim == 3:
        (zstart, ystart, xstart), (zstop, ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
        return (zstart, ystart, xstart), (zstop, ystop, xstop)
    elif A.ndim == 2:
        (ystart, xstart), (ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
        return (ystart, xstart), (ystop, xstop)
    else:
        print('box err')
        return
def focal_loss(y_true, y_pred):
   gamma = 2.
   alpha = .36
   pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
   pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
   return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def sensitivity(y_true, y_pred):
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # return true_positives / (possible_positives + K.epsilon())
    TP = tf.reduce_sum(y_true[:, 1] * tf.round(y_pred[:, 1]))
    TN = tf.reduce_sum((1 - y_true[:, 1]) * (1 - tf.round(y_pred[:, 1])))
    FP = tf.reduce_sum((1 - y_true[:, 1]) * tf.round(y_pred[:, 1]))
    FN = tf.reduce_sum(y_true[:, 1] * (1 - tf.round(y_pred[:, 1])))
    sen=TP/(TP + FN + K.epsilon())
    return sen


def specificity(y_true, y_pred):
    # true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    # possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    # return true_negatives / (possible_negatives + K.epsilon())

    TP = tf.reduce_sum(y_true[:,1] * tf.round(y_pred[:,1]))
    TN = tf.reduce_sum((1 - y_true[:,1]) * (1 - tf.round(y_pred[:,1])))
    FP = tf.reduce_sum((1 - y_true[:,1]) * tf.round(y_pred[:,1]))
    FN = tf.reduce_sum(y_true[:,1] * (1 - tf.round(y_pred[:,1])))
    spec = TN / (TN + FP + K.epsilon())
    return spec

def RScalculate(flists):
    results = []
    for entry in flists:  # Loop over all columns (i.e. the test cases)
        imageFilepath = flists[entry]['Image']
        maskFilepath = flists[entry]['Mask']
        try:
            # print(imageFilepath)
            array, optionsimg = nrrd.read(imageFilepath)
            img_array = sitk.ReadImage(imageFilepath)
            img_array = sitk.GetArrayFromImage(img_array)
            seg_array = sitk.ReadImage(maskFilepath)
            seg_array = sitk.GetArrayFromImage(seg_array)
            img_array = img_array[0,:,:,:]
            seg_array = seg_array[0,:,:]

            (ystart, xstart), (ystop, xstop) = _boundingBox(seg_array)

            ysize = ystop - ystart
            xsize = xstop - xstart

            ycenter = (ystop + ystart) // 2
            xcenter = (xstop + xstart) // 2
            #

            #
            cropSize = max(ysize, xsize)
            deltas = 0


            ystartUse = ycenter - cropSize // 2 - deltas
            ystartUse = max(0,ystartUse)
            ystopUse = ycenter + cropSize // 2 + deltas
            ystopUse = min(ystopUse,np.shape(seg_array)[0])
            xstartUse = xcenter - cropSize // 2 - deltas
            xstartUse = max(0, xstartUse)
            xstopUse = xcenter + cropSize // 2 + deltas
            xstopUse = min(xstopUse, np.shape(seg_array)[1])

            maskUse = seg_array[ystartUse:ystopUse, xstartUse:xstopUse]
            tumorUse = img_array[ystartUse:ystopUse, xstartUse:xstopUse,0:3]

            tumorUse = zoom(tumorUse, (128. / tumorUse.shape[0], 128. / tumorUse.shape[1],1), order=3)

            tumorUse = (tumorUse - np.mean(tumorUse)) / np.std(tumorUse)

            test = np.expand_dims(tumorUse, 0)
            predict = model.predict(test, verbose=0)
            results = np.append(results,predict[0][1])
        except Exception:
            results = np.append(results, 10000)

    return results

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
if __name__ == "__main__":

    outPath = r''
    model = load_model('./model/VPIdeepmodel.hdf5',   custom_objects={'DropBlock2Dmw': DropBlock2Dmw, 'focal_loss': focal_loss,   'sensitivity': sensitivity, 'specificity': specificity})

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

    ID = list(set(IDs))

    iter = 0
    allaccuracy = []
    while iter<=2:
        print ("This is Iter "+ str(iter+1))
        np.random.seed(seed=2*iter)
        selectpind = np.random.randint(0, np.shape(ID)[0], size=(30, 1))
        IDselect=[]
        for entry in selectpind:
            IDselect = np.append(IDselect,ID[entry[0]])

        allRS = [];
        ytrue = [];
        allinfo = pd.DataFrame(columns=['mrn','predictRS','Label'])
        count=0
        for entry in IDselect:
            print("patient " + str(entry)+ " is processing..." )
            ind = np.where(IDs == entry)
            flists2 = flists[ind[0]]
            RS = RScalculate(flists2)
            nRS = combinepatch(RS)
            inc = np.where(IDclinics == entry)
            nRS2 = 1 / (1 + math.exp(-8 * nRS + 0.166 * round(diameters[inc][0]) - 0.909 * noduletypes[inc][0] + 5.925))
            allRS = np.append(allRS, nRS2)
            ytrue = np.append(ytrue, Labels[inc][0])
            allinfo.loc[count] = [[entry], [nRS2], [Labels[inc][0]]]
            count =count+1


        score = copy.deepcopy(allRS)
        score[allRS>=0.5433] = 1
        score[allRS<0.5433] = 0

        accuracy = metrics.accuracy_score(ytrue, score)
        print(allinfo)
        # print("Iter "+ str(iter+1)+ ": accuracy = " + str(100*accuracy)+"%")
        iter = iter+1
        allaccuracy = np.append(allaccuracy,accuracy)

    print("Iter 1 accuracy = " + str(100 * allaccuracy[0]) + "%")
    print("Iter 2 accuracy = " + str(100 * allaccuracy[1]) + "%")
    print("Iter 3 accuracy = " + str(100 * allaccuracy[2]) + "%")
    print("average accuracy = " + str(100*np.mean(allaccuracy))+"%")



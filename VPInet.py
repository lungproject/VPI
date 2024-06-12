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



if __name__ == "__main__":

    outPath = r''
    model = load_model('./model/VPIdeepmodel.hdf5',   custom_objects={'DropBlock2Dmw': DropBlock2Dmw, 'focal_loss': focal_loss,   'sensitivity': sensitivity, 'specificity': specificity})

    inputCSV = os.path.join(outPath, './data/testCases.csv')
    flists = pandas.read_csv(inputCSV).T

    results=[]
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
            # plt.figure()
            # plt.imshow(img_array, interpolation='nearest')
            # plt.show()
            # plt.figure()
            # plt.imshow(seg_array, interpolation='nearest')
            # plt.show()


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


            # plt.figure()
            # plt.imshow(tumorUse, interpolation='nearest')
            # plt.show()

            test = np.expand_dims(tumorUse, 0)
            predict = model.predict(test, verbose=1)
            results = np.append(results,predict[0][1])
        except Exception:
            results = np.append(results, 10000)
    pathtest = "./results/predictRS.npy"
    outfile_x = open(pathtest, 'wb')
    np.save(outfile_x, results)
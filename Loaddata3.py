import os  
# from PIL import Image  
import csv
import numpy as np  
from keras import backend as K
# import scipy.io
# from scipy.io import loadmat

def load_alldata():
    img = np.load("H:/Data/PleuralInvasion/NPY3selorg/allpos.npy")
    x_pos = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3selorg/allneg.npy")
    x_neg = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3selorg/allposstr.npy")
    pos_str = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3selorg/allnegstr.npy")
    neg_str = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3selorg/allposname.npy")
    pos_name = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3selorg/allnegname.npy")
    neg_name = np.asarray(img, dtype="float32")

    return x_pos, pos_str, pos_name, x_neg, neg_str, neg_name

def load_data(): #ct window
     
    img = np.load("H:/Data/PleuralInvasion/NPY3sel/xtrainnewsel.npy")
    datatrain = np.asarray(img,dtype="float32")

    
    img = np.load("H:/Data/PleuralInvasion/NPY3sel/ytrainnewsel.npy")
    labeltrain = np.asarray(img,dtype="float32")


    img = np.load("H:/Data/PleuralInvasion/NPY3sel/xvalnewsel.npy")
    dataval = np.asarray(img,dtype="float32")
    #
    img = np.load("H:/Data/PleuralInvasion/NPY3sel/yvalnewsel.npy")
    labelval = np.asarray(img,dtype="float32")

    # img = np.load("H:/Data/PleuralInvasion/NPY3sel/xexternewsel.npy")
    # dataval = np.asarray(img, dtype="float32")
    # #
    # img = np.load("H:/Data/PleuralInvasion/NPY3sel/yexternewsel.npy")
    # labelval = np.asarray(img, dtype="float32")



    return datatrain,labeltrain,dataval,labelval


def load_testdata():

    img = np.load("H:/Data/PleuralInvasion/NPY3sel/xtrainnewsel.npy")
    datatrain = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3sel/ytrainnewsel.npy")
    labeltrain = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3sel/xvalnewsel.npy")
    dataval = np.asarray(img, dtype="float32")
    #
    img = np.load("H:/Data/PleuralInvasion/NPY3sel/yvalnewsel.npy")
    labelval = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3sel/xtestnewsel.npy")
    datatest = np.asarray(img, dtype="float32")

    img = np.load("H:/Data/PleuralInvasion/NPY3sel/ytestnewsel.npy")
    labeltest = np.asarray(img, dtype="float32")




    return datatrain,labeltrain,dataval, labelval, datatest, labeltest

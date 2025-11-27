import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
import random

GT_BOX_DIM=15

def load_dataset(dataset, n_channels):

    #MEC - KAISTI
    if(dataset=="MEC"):
        ao_mask_peaks=np.load("MECpeaks.npz", allow_pickle=True)['ao_mask'] 
        SCG_subjects=np.load("MECsigNorm_multi.npz", allow_pickle=True)['normalized_scg'] 
        ECG_subjects=np.load("MECsigNorm_multi.npz", allow_pickle=True)['normalized_ecg'] 
        if(n_channels==1):
            SCG_subjects = [np.asarray(s)[:, 2] for s in SCG_subjects]

    #CEBS
    if(dataset=="CEBS"):
        ao_mask_peaks=np.load("CEBSpeaks.npz", allow_pickle=True)['ao_mask']
        SCG_subjects=np.load("CEBSsigNorm.npz", allow_pickle=True)['bas_scg_norm']
        ECG_subjects=np.load("CEBSsigNorm.npz", allow_pickle=True)['bas_ecg_norm']


    gt_boxes_AO_subj=[]

    for i in range(len(SCG_subjects)):
        gt_boxes_SCG=np.zeros(len(SCG_subjects[i]))

        for j in range(len(ao_mask_peaks[i])):
            if(ao_mask_peaks[i][j]==1):
                gt_boxes_SCG[int(j-int(GT_BOX_DIM/2)):int(j+int(GT_BOX_DIM/2))]=1

        gt_boxes_AO_subj.append(gt_boxes_SCG)  

    return ECG_subjects, SCG_subjects, gt_boxes_AO_subj, ao_mask_peaks


def create_trainVal_subj(SCG_subjects, gt_boxes_AO_subj, ao_mask_peaks, window_size, sub):

    x_winds=[]
    y_winds_AO=[]
    for n in range(len(SCG_subjects)):
        if n == sub:
            continue
        end=0
        i=0
        for i in range(len(SCG_subjects[n])//window_size):
            start=end
            end=start+window_size
            x_winds.append(SCG_subjects[n][start:end])
            y_winds_AO.append(gt_boxes_AO_subj[n][start:end])

    x_winds=np.array(x_winds)
    y_winds_AO=np.array(y_winds_AO)

    print("Total windows: ", x_winds.shape)

    temp = list(zip(x_winds, y_winds_AO))
    random.shuffle(temp) 
    res1, res2 = zip(*temp)
    x_train_shuffle, y_binary_train_shuffle = list(res1), list(res2)

    #random train/validation split
    x_train = []
    y_binary_train = []
    x_val = []
    y_val = []
    for i in range(len(x_train_shuffle)):
        if i % 10 == 0:
            x_val.append(x_train_shuffle[i])
            y_val.append(y_binary_train_shuffle[i])
        else:
            x_train.append(x_train_shuffle[i])
            y_binary_train.append(y_binary_train_shuffle[i])

    return np.array(x_train), np.array(y_binary_train), np.array(x_val), np.array(y_val)


def create_trainVal(SCG_subjects, gt_boxes_AO_subj, ao_mask_peaks, window_size):

    x_winds=[]
    y_winds_AO=[]
    for n in range(len(SCG_subjects)):
        end=0
        i=0
        for i in range(len(SCG_subjects[n])//window_size):
            start=end
            end=start+window_size
            x_winds.append(SCG_subjects[n][start:end])
            y_winds_AO.append(gt_boxes_AO_subj[n][start:end])

    x_winds=np.array(x_winds)
    y_winds_AO=np.array(y_winds_AO)

    print("Total windows: ", x_winds.shape)

    temp = list(zip(x_winds, y_winds_AO))
    random.shuffle(temp) 
    res1, res2 = zip(*temp)
    x_train_shuffle, y_binary_train_shuffle = list(res1), list(res2)

    #random train/validation split
    x_train = []
    y_binary_train = []
    x_val = []
    y_val = []
    for i in range(len(x_train_shuffle)):
        if i % 10 == 0:
            x_val.append(x_train_shuffle[i])
            y_val.append(y_binary_train_shuffle[i])
        else:
            x_train.append(x_train_shuffle[i])
            y_binary_train.append(y_binary_train_shuffle[i])

    return np.array(x_train), np.array(y_binary_train), np.array(x_val), np.array(y_val)
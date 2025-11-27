import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import skimage.morphology as sk

def split_windows(arr, arr_ecg, gt, gt_bb, window_size):
    windows_list=[]
    windows_ecg_list=[]
    label_list=[]
    label_list_bb=[]

    index=0
    while index < len(arr)-window_size:
        windows_list.append(arr[index:index+window_size])
        windows_ecg_list.append(arr_ecg[index:index+window_size])
        label_list.append(gt[index:index+window_size])
        label_list_bb.append(gt_bb[index:index+window_size])
        index+=window_size
        index += 1

    return np.array(windows_list), np.array(windows_ecg_list), np.array(label_list), np.array(label_list_bb)



def threshold_pred(pred, thresh):
    pred_thresh=np.zeros(len(pred))

    for i in range(len(pred)):
        if(pred[i]>thresh):
            pred_thresh[i]=1

    return pred_thresh


def validation(pred_new, pred_grad_pos, gt):
    tp=0
    fp=0
    fn=0


    for ao_peak in gt:
        if(pred_new[int(ao_peak)]==1):
            tp+=1
        else:
            fn+=1

    fp=len(pred_grad_pos)-tp
       
    return tp, fp, fn 

def find_rising_edges(binary):

    train_grad=np.gradient(binary)
    train_grad_pos=np.where(train_grad>0)

    train_grad_pos_correct=[]
    for i in range(0, len(train_grad_pos[0]), 2):
        train_grad_pos_correct.append(train_grad_pos[0][i])

    return train_grad_pos_correct


def results(pred, gt, gt_mask, thresh, area_open):

    pred_thresh=threshold_pred(pred, thresh)
    pred_open=sk.area_opening(pred_thresh, area_open)
    pred_peaks = find_rising_edges(pred_open)    
    tp, fp, fn= validation(pred_open, pred_peaks, gt)

    return pred_open, tp, fp, fn


def test_model(model, test_subj, test_y_subj, y_test_mask_peaks, threshold=0.5, area_open=5):

    pred_test=model.predict(test_subj)

    pred_open_list=[]
    tp_list=[]
    fp_list=[]
    fn_list=[]
    prec_list=[]
    recall_list=[]
    f1_list=[]

    for i in range(pred_test.shape[0]):

        AO_ecg_subj=np.argwhere(y_test_mask_peaks[i]==1) 

        pred_open, tp, fp, fn= results(pred_test[i], AO_ecg_subj, test_y_subj[i], threshold, area_open)

        #handle division by zero
        if(tp+fp)==0:
            prec=0
        else:
            prec=tp/(tp+fp)
        if(tp+fn)==0:
            recall=0
        else:
            recall=tp/(tp+fn)
        if(2*tp +fp +fn)==0:
            f1=0
        else:
            f1=2*tp/(2*tp +fp +fn)

        pred_open_list.append(pred_open)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        prec_list.append(prec)
        recall_list.append(recall)
        f1_list.append(f1)


    tp_mean=np.mean(tp_list)
    fp_mean=np.mean(fp_list)
    fn_mean=np.mean(fn_list)
    prec_mean=np.mean(prec_list)
    recall_mean=np.mean(recall_list)
    f1_mean=np.mean(f1_list)

    return pred_open_list, tp_mean, fp_mean, fn_mean, prec_mean, recall_mean, f1_mean



import sys
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'

from tensorflow import keras
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import random

import load_data as ld
import model_functions as mf
import evaluation as ev
import time

import random
import train_subjects as ts





def run_exp(dataset_train, dataset_test, window_size, n_channels, pers, ft):

    print("Training dataset: ", dataset_train)
    print("Testing dataset: ", dataset_test)
    print("Window size: ", window_size)
    print("Number of channels: ", n_channels)
    print("Personalization: ", pers)
    print("Fine-tuning: ", ft)

    time_start=time.time()

    ECG_subjects_server, SCG_subjects_server, gt_boxes_AO_subj_server, ao_mask_peaks_server = ld.load_dataset(dataset_train, n_channels)

    print(np.array(SCG_subjects_server).shape, np.array(gt_boxes_AO_subj_server).shape, np.array(ao_mask_peaks_server).shape, np.array(ECG_subjects_server).shape)

    print("Number of subjects in training dataset: ", len(SCG_subjects_server))
    
    # Train models excluding one subject and save them.
    # The model trained on all subjects but i is saved as model<dataset>_subj_<i>

    print("Training models for each subject...")
    ts.train_subjects(SCG_subjects_server, gt_boxes_AO_subj_server, ao_mask_peaks_server, window_size, n_channels, dataset_train)

    time_end=time.time()
    print("Total training time (s): ", time_end-time_start)

    #evaluate on test set
    if(dataset_test!=dataset_train):

        ECG_subjects_test, SCG_subjects_test, gt_boxes_AO_subj_test, ao_mask_peaks_test = ld.load_dataset(dataset_test, n_channels)


        #select n random indexes from subjects test set 
        n_test_subj=10
        test_subj_indexes=random.sample(range(len(SCG_subjects_test)), n_test_subj)

        ecg_test_finetune=[]
        scg_test_finetune=[]
        gt_boxes_test_finetune=[]
        ao_mask_peaks_test_finetune=[]

        for i in test_subj_indexes:
            ecg_test_finetune.append(ECG_subjects_test[i])
            scg_test_finetune.append(SCG_subjects_test[i])
            gt_boxes_test_finetune.append(gt_boxes_AO_subj_test[i])
            ao_mask_peaks_test_finetune.append(ao_mask_peaks_test[i])


        #select a model between the ones trained on the training dataset
        sub_train=np.random.randint(0, len(SCG_subjects_server))
        print("Evaluating model trained on subject ", sub_train, " of training dataset.")
        model=keras.models.load_model("saved_models/model" + dataset_train + "_subj_"+ str(sub_train))
        window_test_size=640 #dimension of window for evaluation (multiple of 64)

        pred_open_list=[]
        tp_list=[]
        fp_list=[]
        fn_list=[]
        prec_list=[]
        recall_list=[]
        f1_list=[]

        for sub in range(len(scg_test_finetune)):

            x_test_shuffle, wind_ecg, y_binary_mask_test_shuffle, y_binary_test_shuffle=ev.split_windows(scg_test_finetune[sub], ecg_test_finetune[sub], ao_mask_peaks_test_finetune[sub], gt_boxes_test_finetune[sub], window_test_size)

            pred_open, tp, fp, fn, prec, recall, f1= ev.test_model(model, x_test_shuffle, y_binary_test_shuffle, y_binary_mask_test_shuffle, threshold=0.5, area_open=5)

            pred_open_list.append(pred_open)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            prec_list.append(prec)
            recall_list.append(recall)
            f1_list.append(f1)

        print("Average results on test set:")
        print("Precision: ", np.mean(prec_list))
        print("Recall: ", np.mean(recall_list))
        print("F1-score: ", np.mean(f1_list))
        print("\n")

        if(ft==1):

            print("Fine-tuning the model on the other subjects of the test dataset...")

            scg_finetune_subj=[]
            finetune_gt_boxes_AO_subj=[]
            finetune_ao_mask_peaks=[]

            for i in range(len(SCG_subjects_test)):
                if i not in test_subj_indexes:
                    scg_finetune_subj.append(SCG_subjects_test[i])
                    finetune_gt_boxes_AO_subj.append(gt_boxes_AO_subj_test[i])
                    finetune_ao_mask_peaks.append(ao_mask_peaks_test[i])

            modelFT, historyFT = ts.finetune_subjects(model, scg_finetune_subj, finetune_gt_boxes_AO_subj, finetune_ao_mask_peaks, window_size, n_channels, dataset_train, dataset_test, sub_train)

            for sub in range(len(scg_test_finetune)):

                x_test_shuffle, wind_ecg, y_binary_mask_test_shuffle, y_binary_test_shuffle=ev.split_windows(scg_test_finetune[sub], ecg_test_finetune[sub], ao_mask_peaks_test_finetune[sub], gt_boxes_test_finetune[sub], window_test_size)

                pred_open, tp, fp, fn, prec, recall, f1= ev.test_model(modelFT, x_test_shuffle, y_binary_test_shuffle, y_binary_mask_test_shuffle, threshold=0.5, area_open=5)

                pred_open_list.append(pred_open)
                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)
                prec_list.append(prec)
                recall_list.append(recall)
                f1_list.append(f1)

            print("Average results on test set after fine-tuning:")
            print("Precision: ", np.mean(prec_list))
            print("Recall: ", np.mean(recall_list))
            print("F1-score: ", np.mean(f1_list))
            print("\n")

    else:
        window_test_size=640 #dimension of window for evaluation (multiple of 64)

        pred_open_list=[]
        tp_list=[]
        fp_list=[]
        fn_list=[]
        prec_list=[]
        recall_list=[]
        f1_list=[]

        pers_pred_open_list=[]
        pers_tp_list=[]
        pers_fp_list=[]
        pers_fn_list=[]
        pers_prec_list=[]
        pers_recall_list=[]
        pers_f1_list=[]

        for sub in range(len(SCG_subjects_server)):
            model=keras.models.load_model("saved_models/model" + dataset_train + "_subj_"+ str(sub))

            x_test_shuffle, wind_ecg, y_binary_mask_test_shuffle, y_binary_test_shuffle=ev.split_windows(SCG_subjects_server[sub], ECG_subjects_server[sub], ao_mask_peaks_server[sub], gt_boxes_AO_subj_server[sub], window_size)

            #select randomly 10% of windows for personalization
            total_winds=x_test_shuffle.shape[0]
            test_winds=int(0.1*total_winds)
            test_indexes=random.sample(range(0, total_winds), test_winds)
            pers_x_test_shuffle=x_test_shuffle[test_indexes]
            pers_y_binary_mask_test_shuffle=y_binary_mask_test_shuffle[test_indexes]
            pers_y_binary_test_shuffle=y_binary_test_shuffle[test_indexes]

            eval_x_test_shuffle=np.delete(x_test_shuffle, test_indexes, axis=0)
            eval_y_binary_mask_test_shuffle=np.delete(y_binary_mask_test_shuffle, test_indexes, axis=0)
            eval_y_binary_test_shuffle=np.delete(y_binary_test_shuffle, test_indexes, axis=0)

            #evaluate pretrained model
            pred_open, tp, fp, fn, prec, recall, f1= ev.test_model(model, eval_x_test_shuffle, eval_y_binary_test_shuffle, eval_y_binary_mask_test_shuffle, threshold=0.5, area_open=5)

            pred_open_list.append(pred_open)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            prec_list.append(prec)
            recall_list.append(recall)
            f1_list.append(f1)

            if(pers==1):

                print("Personalizing model for subject ", sub)

                modelPers, historyPers = ts.personalize_subjects(model, pers_x_test_shuffle, pers_y_binary_mask_test_shuffle, pers_y_binary_test_shuffle, window_size, n_channels, dataset_train, sub)

                pred_open, tp, fp, fn, prec, recall, f1= ev.test_model(modelPers, eval_x_test_shuffle, eval_y_binary_test_shuffle, eval_y_binary_mask_test_shuffle, threshold=0.5, area_open=5)
                pers_pred_open_list.append(pred_open)
                pers_tp_list.append(tp)
                pers_fp_list.append(fp)
                pers_fn_list.append(fn)
                pers_prec_list.append(prec)
                pers_recall_list.append(recall)
                pers_f1_list.append(f1)

        print("Average results on test set:")
        print("Precision: ", np.mean(prec_list))
        print("Recall: ", np.mean(recall_list))
        print("F1-score: ", np.mean(f1_list))
        print("\n")

        if(pers==1):
            print("Average results on test set after personalization:")
            print("Precision: ", np.mean(pers_prec_list))
            print("Recall: ", np.mean(pers_recall_list))
            print("F1-score: ", np.mean(pers_f1_list))
            print("\n")



if __name__ == "__main__":
    dataset_train = str(sys.argv[1])
    dataset_test = str(sys.argv[2])
    window_size = int(sys.argv[3])
    n_channels=int(sys.argv[4])
    pers=int(sys.argv[5])
    ft=int(sys.argv[6])

    run_exp(dataset_train, dataset_test, window_size, n_channels, pers, ft)

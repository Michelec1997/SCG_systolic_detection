import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
from tensorflow import keras
import numpy as np

import load_data as ld
import model_functions as mf

LR= 10e-4
EPOCHS=100
EPOCHS_FT=30
EPOCHS_PERS=30

BATCH_SIZE=32
VERBOSE=1
PATIENCE=10


def train_subjects(SCG_subjects_server, gt_boxes_AO_subj_server, ao_mask_peaks_server, window_size, n_channels, dataset_train):
    """Train a model for each subject and save them to the folder `saved_models/`.
    """
    time_start = time.time()

    for sub in range(len(SCG_subjects_server)):
        time_start_sub = time.time()

        x_train, y_train, x_val, y_val = ld.create_trainVal_subj(
            SCG_subjects_server, gt_boxes_AO_subj_server, ao_mask_peaks_server, window_size, sub
        )

        #add channel axis if needed
        if x_train.ndim == 2:
            print("Adding channel axis to training and validation data")
            x_train = x_train[..., np.newaxis]
            x_val = x_val[..., np.newaxis]

        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

        model = mf.build_model(None, n_channels)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LR), metrics=['accuracy'])
        #print shape of x_train and y_train
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[callback], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

        model.save("saved_models/model" + dataset_train + "_subj_" + str(sub))

        time_sub = time.time()
        print("Training time for subj ", sub, " (s): ", time_sub - time_start_sub)

    time_end = time.time()
    print("Total training time (s): ", time_end - time_start)

    return model, history


def finetune_subjects(model, scg_finetune_subj, finetune_gt_boxes_AO_subj, finetune_ao_mask_peaks, window_size, n_channels, dataset_train, dataset_test, sub_train):
    """Fine-tune a model using data from subjects of a different dataset.
    """
    time_start = time.time()

    x_train, y_train, x_val, y_val = ld.create_trainVal(
        scg_finetune_subj, finetune_gt_boxes_AO_subj, finetune_ao_mask_peaks, window_size
    )

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LR), metrics=['accuracy'])

    historyFT = model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[callback], epochs=EPOCHS_FT, batch_size=BATCH_SIZE, verbose=VERBOSE)

    model.save("saved_models/model" + dataset_train + "_subj_" + str(sub_train) + "_FT_" + dataset_test)

    time_end = time.time()
    print("Total fine-tuning time (s): ", time_end - time_start)

    return model, historyFT



def personalize_subjects(model, pers_scg_subj, pers_gt_boxes_AO_subj, pers_ao_mask_peaks, window_size, n_channels, dataset_train, sub):
    """Personalize a model using data from the target subject.
    """
    time_start = time.time()

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LR), metrics=['accuracy'])

    historyPers = model.fit(pers_scg_subj, pers_gt_boxes_AO_subj, validation_split=0.1, callbacks=[callback], epochs=EPOCHS_PERS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    model.save("saved_models/model" + dataset_train + "_subj_" + str(sub) + "_PERS")

    time_end = time.time()
    print("Total personalization time (s): ", time_end - time_start)

    return model, historyPers
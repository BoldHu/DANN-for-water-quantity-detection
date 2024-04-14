# read the data and divide it to train set and test set
import pandas as pd
import numpy as np
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split

features_list = ['X_kla120.mat', 
                 'X_kla240.mat',
                 'X_kla360.mat',
                 'X_kla480.mat',
                 'X_mu0.5.mat',
                 'X_mu0.7.mat',
                 'X_mu0.9.mat']

labels_list = ['EQvec_kla120.mat',
                'EQvec_kla240.mat',
                'EQvec_kla360.mat',
                'EQvec_kla480.mat',
                'EQvec_mu0.5.mat',
                'EQvec_mu0.7.mat',
                'EQvec_mu0.9.mat']

nums_datasets = len(features_list)

for i in range(nums_datasets):
    # load the data
    data = sio.loadmat(os.path.join('Data', features_list[i]))['data']
    labels = sio.loadmat(os.path.join('Data', labels_list[i]))['EQvec']
    # divide the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # save the data
    sio.savemat(os.path.join('Data', 'train_' + features_list[i]), {'data': X_train})
    sio.savemat(os.path.join('Data', 'train_' + labels_list[i]), {'EQvec': y_train})
    sio.savemat(os.path.join('Data', 'test_' + features_list[i]), {'data': X_test})
    sio.savemat(os.path.join('Data', 'test_' + labels_list[i]), {'EQvec': y_test})
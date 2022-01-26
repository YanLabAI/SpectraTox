"""
Created on 2021/12/06

@author: hu song
"""
import pandas as pd
import numpy as np
import copy
import os

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import ML_Regress_auto, DL_Regress_auto, ML_Classifier_auto, DL_Classifier_auto
from models import support

T_f = DL_Regress_auto.T_f
batch_size = 16

def train_IGC50():
    ## train ML
    train = np.array(pd.read_excel('./data/IGC50_train.xlsx', index_col=None).iloc[:,5:])
    train_tox = np.array(pd.read_excel('./data/IGC50_train.xlsx', index_col=None).iloc[:,4])
    test = np.array(pd.read_excel('./data/IGC50_test.xlsx', index_col=None).iloc[:,5:])
    test_tox = np.array(pd.read_excel('./data/IGC50_test.xlsx', index_col=None).iloc[:,4])

    data_all = np.concatenate((train, test), axis=0)
    len_train = len(train)
    train = StandardScaler().fit_transform(data_all)[:len_train,:]
    test = StandardScaler().fit_transform(data_all)[len_train:,:]

    svr_model, svr_kf_pred, svr_test_pred = ML_Regress_auto.svr_auto(train, train_tox, test, test_tox, 1, 5, 'IGC50', 'SVM')
    rfr_model, rfr_kf_pred, rfr_test_pred = ML_Regress_auto.rfr_auto(train, train_tox, test, test_tox, 1, 5, 'IGC50', 'RF')
    br_model, br_kf_pred, br_test_pred = ML_Regress_auto.br_auto(train, train_tox, test, test_tox, 1, 5, 'IGC50', 'BRR')
    xgb_model, xgb_kf_pred, xgb_test_pred = ML_Regress_auto.xgbr_auto(train, train_tox, test, test_tox, 1, 5, 'IGC50', 'XGB')

    ## train FCNN
    train_dataset = [(T_f(train[i,:]), T_f(train_tox[i].reshape(-1))) for i in range(len(train_tox))]
    test_dataset = [(T_f(test[i,:]), T_f(test_tox[i].reshape(-1))) for i in range(len(test_tox))]
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=False)#,num_workers=workers)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False)
    model = DL_Regress_auto.Net_FCNN_r(393)

    scores = DL_Regress_auto.cross_val_regress(model, 1000, train_dataset, 1, 5)
    FCNN_test_mse, FCNN_test_r2, FCNN_test_pred, FCNN_test_obs = DL_Regress_auto.train_Net_r(trainloader, testloader, 1000, model)

    support.result_save_regressor(train_tox, scores.pred_all, test_tox, FCNN_test_pred, "IGC50", "FCNN", "./results")

    ## train CNN
    train_dataset = [(T_f(train[i,:].reshape(1,393)), T_f(train_tox[i].reshape(-1))) for i in range(len(train_tox))]
    test_dataset = [(T_f(test[i,:].reshape(1,393)), T_f(test_tox[i].reshape(-1))) for i in range(len(test_tox))]
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=False)#,num_workers=workers)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False)
    model = DL_Regress_auto.Net_CNN_r()

    scores = DL_Regress_auto.cross_val_regress(model, 1000, train_dataset, 1, 5)
    CNN_test_mse, CNN_test_r2, CNN_test_pred, CNN_test_obs = DL_Regress_auto.train_Net_r(trainloader, testloader, 1000, model)

    support.result_save_regressor(train_tox, scores.pred_all, test_tox, CNN_test_pred, "IGC50", "CNN", "./results")

def train_Liver():
    train = np.array(pd.read_excel('./data/Liver_train.xlsx', index_col=None).iloc[:,5:])
    train_tox = np.array(pd.read_excel('./data/Liver_train.xlsx', index_col=None).iloc[:,4])
    test = np.array(pd.read_excel('./data/Liver_test.xlsx', index_col=None).iloc[:,5:])
    test_tox = np.array(pd.read_excel('./data/Liver_test.xlsx', index_col=None).iloc[:,4])

    data_all = np.concatenate((train, test), axis=0)
    len_train = len(train)
    train = StandardScaler().fit_transform(data_all)[:len_train,:]
    test = StandardScaler().fit_transform(data_all)[len_train:,:]

    svr_model, svr_kf_pred, svr_test_pred = ML_Classifier_auto.svc_auto(train, train_tox, test, test_tox, 1, 5, 'IGC50', 'SVM')
    rfr_model, rfr_kf_pred, rfr_test_pred = ML_Classifier_auto.rfc_auto(train, train_tox, test, test_tox, 1, 5, 'IGC50', 'RF')
    knn_model, knn_kf_pred, knn_test_pred = ML_Classifier_auto.knnc_auto(train, train_tox, test, test_tox, 1, 5, 'IGC50', 'kNN')
    xgb_model, xgb_kf_pred, xgb_test_pred = ML_Classifier_auto.xgbc_auto(train, train_tox, test, test_tox, 1, 5, 'IGC50', 'XGB')

    ## Liver FCNN
    batch_size = 16
    train_dataset = [(T_f(train[i, :]), T_f(train_tox[i], torch.long)) for i in range(len(train_tox))]
    test_dataset = [(T_f(test[i, :]), T_f(test_tox[i], torch.long)) for i in range(len(test_tox))]
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                              shuffle=False)  # ,num_workers=workers)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = DL_Classifier_auto.Net_FCNN_c(1000)

    scores = DL_Classifier_auto.cross_val_Classifier(model, 1000, train_dataset, 1, 5)
    FCNN_test_mse, FCNN_test_r2, FCNN_test_pred, FCNN_test_obs = DL_Classifier_auto.train_Net_c(trainloader, testloader, 1000, model)

    support.result_save_classifier(train_tox, scores.pred_all, test_tox, FCNN_test_pred, "Liver", "FCNN", "./results")

    ## Liver CNN
    train_dataset = [(T_f(train[i, :]).reshape(1, 1000), T_f(train_tox[i], torch.long)) for i in
                     range(len(train_tox))]
    test_dataset = [(T_f(test[i, :]).reshape(1, 1000), T_f(test_tox[i], torch.long)) for i in
                    range(len(test_tox))]
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                              shuffle=False)  # ,num_workers=workers)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = DL_Classifier_auto.Net_CNN_c()

    scores = DL_Classifier_auto.cross_val_Classifier(model, 1000, train_dataset, 1, 5)
    CNN_test_mse, CNN_test_r2, CNN_test_pred, CNN_test_obs = DL_Classifier_auto.train_Net_c(trainloader, testloader, 1000, model)

    support.result_save_classifier(train_tox, scores.pred_all, test_tox, CNN_test_pred, "Liver", "CNN", "./results")

if __name__ == '__main__':
    train_IGC50()
    train_Liver()

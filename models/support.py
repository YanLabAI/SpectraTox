import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score

def AUC(true, pred):
    if np.sum(true)==len(true) or np.sum(true)==0:
        return "ALL num are same"
    else:
        return auc(true, pred)

def result_save_regressor(y_test_train, pred_train, y_test_test, pred_test, data_neme, model_name, path):
    # kf_true, kf_pred, test_true, test_pred
    os.makedirs("{}/figure".format(path), exist_ok=True)
    os.makedirs("{}/csv".format(path), exist_ok=True)
    
    r2_train = r2_score(y_test_train, pred_train)
    r2_test = r2_score(y_test_test, pred_test)

    mse_train = MSE(y_test_train, pred_train)
    mse_test = MSE(y_test_test, pred_test)

    mae_train = MAE(y_test_train, pred_train)
    mae_test = MAE(y_test_test, pred_test)
    
    pd.concat([pd.DataFrame(pred_train,columns=["prediction"])
               ,pd.DataFrame(y_test_train,columns=['true'])
               ,pd.DataFrame([r2_train],columns=["r^2"])
               ,pd.DataFrame([np.sqrt(mse_train)],columns=['RMSE'])
               ,pd.DataFrame([mae_train],columns=['MAE'])],axis =1).to_csv("{}/csv/{}_{}_kf.csv".format(path, data_neme, model_name), index=None)
    
    pd.concat([pd.DataFrame(pred_test,columns=["prediction"])
               ,pd.DataFrame(y_test_test,columns=['true'])
               ,pd.DataFrame([r2_test],columns=["r^2"])
               ,pd.DataFrame([np.sqrt(mse_test)],columns=['RMSE'])
               ,pd.DataFrame([mae_test],columns=['MAE'])],axis =1).to_csv("{}/csv/{}_{}_test.csv".format(path, data_neme, model_name), index=None)
    
    print("R2-5cv = {:.3f} R2-val = {:.3f}".format(r2_train, r2_test))
    print("RMSE-5cv = {:.3f} RMSE-val = {:.3f}".format(np.sqrt(mse_train), np.sqrt(mse_test)))
    print("MAE-5cv = {:.3f} MAE-val = {:.3f}".format(mae_train, mae_test))
    
def result_save_classifier(y_test_train, pred_train, y_test_test, pred_test, data_neme, model_name, path):
    # kf_true, kf_pred, test_true, test_pred
    # os.makedirs("{}/figure".format(path), exist_ok=True)
    os.makedirs("{}/csv".format(path), exist_ok=True)

    r2_train = accuracy_score(y_test_train, pred_train)
    r2_test = accuracy_score(y_test_test, pred_test)

    mse_train = f1_score(y_test_train, pred_train)
    mse_test = f1_score(y_test_test, pred_test)

    mae_train = AUC(y_test_train, pred_train)
    mae_test = AUC(y_test_test, pred_test)

    pd.concat([pd.DataFrame(pred_train, columns=["prediction"])
                  , pd.DataFrame(y_test_train, columns=['true'])
                  , pd.DataFrame([r2_train], columns=["ACC"])
                  , pd.DataFrame([mse_train], columns=['f1_score'])
                  , pd.DataFrame([mae_train], columns=['AUC'])], axis=1).to_csv(
        "{}/csv/{}_{}_kf.csv".format(path, data_neme, model_name), index=None)

    pd.concat([pd.DataFrame(pred_test, columns=["prediction"])
                  , pd.DataFrame(y_test_test, columns=['true'])
                  , pd.DataFrame([r2_test], columns=["ACC"])
                  , pd.DataFrame([mse_test], columns=['f1_score'])
                  , pd.DataFrame([mae_test], columns=['AUC'])], axis=1).to_csv(
        "{}/csv/{}_{}_test.csv".format(path, data_neme, model_name), index=None)

    print("ACC-5cv = {:.3f} ACC-val = {:.3f}".format(r2_train, r2_test))
    print("f1_score-5cv = {:.3f} f1_score-val = {:.3f}".format(mse_train, mse_test))
    print("AUC-5cv = {} AUC-val = {}".format(mae_train, mae_test))
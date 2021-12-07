import pandas as pd
import os
import numpy as np
import copy

from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.svm import SVR as svr
from sklearn.linear_model import BayesianRidge as br
from xgboost import XGBRegressor as xgbr

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.cm as cm
import matplotlib

class cross_val():
    def __init__(self, model, X, y, random_state, cv):
        super(cross_val, self).__init__()
        
        if random_state != 0:
            index = np.arange(0,len(y))
            np.random.seed(random_state)
            np.random.shuffle(index)
            np.random.seed(random_state)
            np.random.shuffle(index)
        else:
            index = np.arange(len(y))

        step = int(len(y)/cv)
        self.r2 = []
        self.mse = []
        self.pred_all = np.zeros((len(y)), dtype=float)
        self.obs_all = np.zeros((len(y)), dtype=float)
        for i in range(cv):
            if i < cv-1:
                index_train = np.concatenate([index[:i*step],index[(i+1)*step:]], axis=0)
                index_val = index[i*step:(i+1)*step]
            else: 
                index_train = index[0:i*step]
                index_val = index[i*step:]
        
            X_train = X[index_train]
            y_train = y[index_train]
            X_val = X[index_val]
            y_val = y[index_val]
            
            pred = model.fit(X_train,y_train).predict(X_val)
#             self.pred_all = np.concatenate([self.pred_all,pred], axis=0)
            self.pred_all[index_val] = pred
            self.obs_all[index_val] = y_val
            self.r2.append(r2_score(y_val, pred))
            self.mse.append(MSE(y_val, pred))
            
        self.r2_mean = r2_score(self.obs_all, self.pred_all)
        self.mse_mean = MSE(self.obs_all, self.pred_all)

def result_figure(y_test_train, pred_train, y_test_test, pred_test, data_neme, model_name, path):
    # kf_true, kf_pred, test_true, test_pred
    # os.makedirs("{}/figure".format(path), exist_ok=True)
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
    
def rfr_auto(X, y, test_X, test_y, random_state, cv, data_name, model_name):

    print("*"*100)
    print(data_name, f"Model:{model_name}")
    
    model = rfr(n_jobs=-1)
    before_kf_r2 = cross_val(model, X, y, random_state, cv).r2_mean
    before_kf_mse = cross_val(model, X, y, random_state, cv).mse_mean
    
    test_pred = model.fit(X, y).predict(test_X)
    before_test_r2 = r2_score(test_y, test_pred)
    before_test_mse = MSE(test_y, test_pred)
    
    print("before kf r^2", before_kf_r2)
    print("before kf mse", before_kf_mse)
    print("before test r^2", before_test_r2)
    print("before test mse", before_test_mse)
    
    # 随机种子
    scores = []
    for i in range(0,200):
        model = rfr(n_jobs=-1
                    ,random_state=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass
    
    plt.figure()
    plt.plot(range(0,200),scores)
    plt.show()

    random_state_rf = range(0,200)[scores.index(max(scores))]
    print("random_state:", max(scores),random_state_rf)
    
    # 随机树数目
    scores = []
    for i in range(1,200):
        model = rfr(n_estimators=i
                   ,random_state=random_state_rf
                   ,n_jobs=-1)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass    
    

    plt.figure()
    plt.plot(range(1,200),scores)
    plt.show()

    n_estimators = range(1,200)[scores.index(max(scores))]
    print("n_estimators:", max(scores),n_estimators)
    
    # 最大深度
    scores = []
    for i in range(1,200):
        model = rfr(n_estimators=n_estimators
                   ,random_state=random_state_rf
                   ,n_jobs=-1
                   ,max_depth=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass            

    plt.figure()
    plt.plot(range(1,200),scores)
    plt.show()

    max_depth = range(1,200)[scores.index(max(scores))]
    print("max_depth", max(scores),max_depth)
    
    # 最大特征选择
    scores = []
    for i in range(int(X.shape[1]**0.5),X.shape[1]):
        model = rfr(n_jobs=-1
                   ,n_estimators=n_estimators
                   ,random_state=random_state_rf
                   ,max_depth=max_depth
                   ,max_features=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass

    plt.figure()
    plt.plot(range(int(X.shape[1]**0.5),X.shape[1]),scores)
    plt.show()

    max_features = range(int(X.shape[1]**0.5),X.shape[1])[scores.index(max(scores))]
    print("max_features", max(scores),max_features)
    
    # 最小纯度递减
    scores = []
    for i in np.linspace(0,0.5,20):
        model = rfr(n_jobs=-1
                   ,n_estimators=n_estimators
                   ,random_state=random_state_rf
                   ,max_depth=max_depth
                   ,max_features=max_features
                   ,min_impurity_decrease=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass

    plt.figure()
    plt.plot(np.linspace(0,0.5,20),scores)
    plt.show()

    min_impurity_decrease = np.linspace(0,0.5,20)[scores.index(max(scores))]
    print("min_impurity_decrease:", max(scores),min_impurity_decrease)
    
    # 最大样本数量
    scores = []
    len_train = int(len(y)/cv)*(cv-1)+1
    for i in range(1,len_train):
        model = rfr(n_jobs=-1
                   ,n_estimators=n_estimators
                   ,random_state=random_state_rf
                   ,max_depth=max_depth
                   ,max_features=max_features
                   ,min_impurity_decrease=min_impurity_decrease
                   ,max_samples=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass

    plt.figure()
    plt.plot(range(1,len_train),scores)
    plt.show()

    max_samples = range(1,len_train)[scores.index(max(scores))]
    print("max samples", max(scores),max_samples)
    
    ####################################
#     n_estimators= 110
#     random_state_rf= 102
#     max_depth= 17
#     max_features= 435
#     min_impurity_decrease= 0.0
#     max_samples= 828

    model = rfr(n_jobs=-1
               ,n_estimators=n_estimators
               ,random_state=random_state_rf
               ,max_depth=max_depth
               ,max_features=max_features
               ,min_impurity_decrease=min_impurity_decrease
               ,max_samples=max_samples)
    score = cross_val(model, X, y, random_state, cv)
    after_kf_r2 = score.r2_mean
    after_kf_mse = score.mse_mean
    
    # 重新根据max_sample的大小范围等比例放缩
    max_samples = round(max_samples/len_train*len(y))
#     max_samples= 1037
    model = rfr(n_jobs=-1
               ,n_estimators=n_estimators
               ,random_state=random_state_rf
               ,max_depth=max_depth
               ,max_features=max_features
               ,min_impurity_decrease=min_impurity_decrease
               ,max_samples=max_samples)
    test_pred = model.fit(X, y).predict(test_X)
    after_test_r2 = r2_score(test_y, test_pred)
    after_test_mse = MSE(test_y, test_pred)
    
    print("kf r^2:", before_kf_r2, "->", after_kf_r2)
    print("kf MSE:", before_kf_mse, "->", after_kf_mse)
    
    print("test r^2:", before_test_r2, "->", after_test_r2)
    print("test MSE:", before_test_mse, "->", after_test_mse)

    
    print("n_estimators=",n_estimators,","
          , "random_state=",random_state_rf,","
          , "max_depth=",max_depth,","
          , "max_features=",max_features,","
          , "min_impurity_decrease=",min_impurity_decrease,","
          , "max_samples=",max_samples)
    
    os.makedirs("./model", exist_ok = True)
    pd.DataFrame([n_estimators
                  , random_state_rf
                  , max_depth
                  , max_features
                  , min_impurity_decrease
                  , max_samples]).to_csv("./model/parma_{}_{}.csv".format(data_name, model_name),index=0)
    
    result_figure(y, score.pred_all, test_y, test_pred, data_name, model_name, "./results")
    return model, score.pred_all, test_pred
    
def svr_auto(X, y, test_X, test_y, random_state, cv, data_name, model_name):

    print("*"*100)
    print(data_name, f"Model:{model_name}")
    
    model = svr()
    before_kf_r2 = cross_val(model, X, y, 1, cv).r2_mean
    before_kf_mse = cross_val(model, X, y, 1, cv).mse_mean
    
    test_pred = model.fit(X, y).predict(test_X)
    before_test_r2 = r2_score(test_y, test_pred)
    before_test_mse = MSE(test_y, test_pred)
    
    print("before kf r^2", before_kf_r2)
    print("before kf mse", before_kf_mse)
    print("before test r^2", before_test_r2)
    print("before test mse", before_test_mse)


    # kernel
    scores = []
    for i in ["rbf", "poly", "sigmoid"]:
        model = svr(kernel=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass

    plt.figure()
    plt.bar(["rbf", "poly", "sigmoid"],scores)
    plt.xticks(ticks=["rbf", "poly", "sigmoid"])
    plt.show()


    kernel = ["rbf", "poly", "sigmoid"][scores.index(max(scores))]
    print(max(scores),kernel)

    # C
    scores = []
    range_list = np.arange(1,50,0.05)
    for i in range_list:
        model = svr(kernel="rbf", C=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)

    plt.figure()
    plt.plot(range_list,scores)
    plt.show()

    C = range_list[scores.index(max(scores))]
    print(max(scores),C)

    # gamma
    scores = []
    range_list = np.linspace(0.0001,100/X.shape[1],200)
    for i in range_list:
        model = svr(kernel=kernel,gamma=i, C=C)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)

    plt.figure()
    plt.plot(range_list,scores)
    plt.show()

    gamma = range_list[scores.index(max(scores))]
    if score.r2_mean > cross_val(svr(kernel=kernel,gamma="scale", C=C), X, y, random_state, cv).r2_mean:
        gamma = gamma
    else:
        gamma = 'scale'
    print(max(scores),gamma)

    ####################################
    model = svr(kernel=kernel,gamma=gamma, C=C)
    score = cross_val(model, X, y, 1, cv)
    after_kf_r2 = score.r2_mean
    after_kf_mse = score.mse_mean
    
    test_pred = model.fit(X, y).predict(test_X)
    after_test_r2 = r2_score(test_y, test_pred)
    after_test_mse = MSE(test_y, test_pred)
    
    print("kf r^2:", before_kf_r2, "->", after_kf_r2)
    print("kf MSE:", before_kf_mse, "->", after_kf_mse)
    
    print("test r^2:", before_test_r2, "->", after_test_r2)
    print("test MSE:", before_test_mse, "->", after_test_mse)
    
    print("kernel=", kernel, ","
          , "C=", C, ","
          , "gamma=", gamma, ",")
    
    result_figure(y, score.pred_all, test_y, test_pred, data_name, model_name, "./results")
    return model, score.pred_all, test_pred

def br_auto(X, y, test_X, test_y, random_state, cv, data_name, model_name):

    print("*"*100)
    print(data_name, f"Model:{model_name}")
    
    model = br(compute_score=True,
                        n_iter=300,
                        alpha_1=1e-06,
                        alpha_2=1e-06,
                        lambda_1=1e-06,
                        lambda_2=1e-06)
    before_kf_r2 = cross_val(model, X, y, random_state, cv).r2_mean
    before_kf_mse = cross_val(model, X, y, random_state, cv).mse_mean
    
    test_pred = model.fit(X, y).predict(test_X)
    before_test_r2 = r2_score(test_y, test_pred)
    before_test_mse = MSE(test_y, test_pred)
    
    print("before kf r^2", before_kf_r2)
    print("before kf mse", before_kf_mse)
    print("before test r^2", before_test_r2)
    print("before test mse", before_test_mse)

    ####################################
    model = br(compute_score=True,
                        n_iter=300,
                        alpha_1=1e-06,
                        alpha_2=1e-06,
                        lambda_1=1e-06,
                        lambda_2=1e-06)
    score = cross_val(model, X, y, random_state, cv)
    after_kf_r2 = score.r2_mean
    after_kf_mse = score.mse_mean
    
    test_pred = model.fit(X, y).predict(test_X)
    after_test_r2 = r2_score(test_y, test_pred)
    after_test_mse = MSE(test_y, test_pred)
    
    print("kf r^2:", before_kf_r2, "->", after_kf_r2)
    print("kf MSE:", before_kf_mse, "->", after_kf_mse)
    
    print("test r^2:", before_test_r2, "->", after_test_r2)
    print("test MSE:", before_test_mse, "->", after_test_mse)
    
    result_figure(y, score.pred_all, test_y, test_pred, data_name, model_name, "./results")
    return model, score.pred_all, test_pred

def xgbr_auto(X, y, test_X, test_y, random_state, cv, data_name, model_name):
    print("*" * 100)
    print(data_name, f"Model:{model_name}")
    model = xgbr(n_jobs=-1, objective='reg:squarederror')
    before_kf_r2 = cross_val(model, X, y, random_state, cv).r2_mean
    before_kf_mse = cross_val(model, X, y, random_state, cv).mse_mean

    test_pred = model.fit(X, y).predict(test_X)
    before_test_r2 = r2_score(test_y, test_pred)
    before_test_mse = MSE(test_y, test_pred)

    print("before kf r^2", before_kf_r2)
    print("before kf mse", before_kf_mse)
    print("before test r^2", before_test_r2)
    print("before test mse", before_test_mse)

    # num_boost and eta
    scores = np.zeros((len(range(1, 101)), len(np.arange(0.01, 0.5, 0.01))), dtype=np.float64)
    score = 0
    imax = 0
    jmax = 0
    for i, n_estimators in enumerate(range(1, 101)):
        for j, eta in enumerate(np.arange(0.01, 0.5, 0.01)):
            model = xgbr(n_jobs=-1, objective='reg:squarederror', eta=eta, n_estimators=n_estimators)
            scores[i][j] = cross_val(model, X, y, random_state, cv).r2_mean

            if scores[i][j] > score:
                score = scores[i][j]
                jmax = j
                imax = i

    fig = plt.figure(figsize=[10, 10])
    ax3d = Axes3D(fig)
    X_ax, y_ax = np.meshgrid(range(1, 101), np.arange(0.01, 0.5, 0.01))
    matrix = np.array(scores.T)
    ax3d.plot_surface(X_ax, y_ax, matrix, linewidth=0, antialiased=False, shade=True, alpha=0.5,
                      cmap='rainbow')  # facecolors=cm.viridis(matrix),cmap=plt.cm.spring)#cmap=plt.cm.spring)#cmap='rainbow')
    plt.show()

    n_estimators = range(1, 101)[imax]
    eta = np.arange(0.01, 0.5, 0.01)[jmax]
    print(score, "n_estimators:", n_estimators, "\n", "eta:", eta)

    # max_depth
    scores = []
    for i in range(1, 50):
        model = xgbr(n_jobs=-1, objective='reg:squarederror', eta=eta, n_estimators=n_estimators,
                              max_depth=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass

    plt.figure()
    plt.plot(range(1, 50), scores)
    plt.show()

    max_depth = range(1, 50)[scores.index(max(scores))]
    print("max_depth:", max(scores), max_depth)

    # gamma
    scores = []
    for i in np.arange(0, 5, 0.5):
        model = xgbr(n_jobs=-1, objective='reg:squarederror', eta=eta, n_estimators=n_estimators,
                              max_depth=max_depth, gamma=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass

    plt.figure()
    plt.plot(np.arange(0, 5, 0.5), scores)
    plt.show()

    gamma = np.arange(0, 5, 0.5)[scores.index(max(scores))]
    print("gamma", max(scores), gamma)

    # reg_alpha
    scores = []
    for i in np.arange(0, 5, 0.05):
        model = xgbr(n_jobs=-1, objective='reg:squarederror', eta=eta, n_estimators=n_estimators,
                              max_depth=max_depth, gamma=gamma, reg_alpha=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass

    plt.figure()
    plt.plot(np.arange(0, 5, 0.05), scores)
    plt.show()

    reg_alpha = np.arange(0, 5, 0.05)[scores.index(max(scores))]
    print("reg_alpha", max(scores), reg_alpha)

    # reg_lambda
    scores = []
    for i in np.arange(0, 5, 0.05):
        model = xgbr(n_jobs=-1, objective='reg:squarederror', eta=eta, n_estimators=n_estimators,
                              max_depth=max_depth, gamma=gamma, reg_alpha=reg_alpha, reg_lambda=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.r2_mean)
        pass

    plt.figure()
    plt.plot(np.arange(0, 5, 0.05), scores)
    plt.show()

    reg_lambda = np.arange(0, 5, 0.05)[scores.index(max(scores))]
    print("reg_lambda:", max(scores), reg_lambda)

    ####################################
    # param = {'objective': 'reg:squarederror'
    #     , 'eta': eta
    #     , 'max_depth': max_depth
    #     , 'gamma': gamma
    #     , 'reg_alpha': reg_alpha
    #     , 'lambda': reg_lambda}
    model = xgbr(n_jobs=-1, objective='reg:squarederror', eta=eta, n_estimators=n_estimators,
                          max_depth=max_depth, gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    score = cross_val(model, X, y, random_state, cv)
    after_kf_r2 = score.r2_mean
    after_kf_mse = score.mse_mean

    test_pred = model.fit(X, y).predict(test_X)
    after_test_r2 = r2_score(test_y, test_pred)
    after_test_mse = MSE(test_y, test_pred)

    print("kf r^2:", before_kf_r2, "->", after_kf_r2)
    print("kf MSE:", before_kf_mse, "->", after_kf_mse)

    print("test r^2:", before_test_r2, "->", after_test_r2)
    print("test MSE:", before_test_mse, "->", after_test_mse)

    print('objective', 'reg:squarederror'
          ,'n_estimators', n_estimators
          , 'eta', eta
          , 'max_depth', max_depth
          , 'gamma', gamma
          , 'reg_alpha', reg_alpha
          , 'reg_lambda', reg_lambda)

    # os.makedirs("./model", exist_ok=True)
    # pd.DataFrame([n_estimators
    #                  , eta
    #                  , max_depth
    #                  , gamma
    #                  , reg_alpha
    #                  , reg_lambda]).to_csv("./model/parma_{}_{}.csv".format(data_name, model_name), index=0)

    result_figure(y, score.pred_all, test_y, test_pred, data_name, model_name, "./results")
    return model, score.pred_all, test_pred
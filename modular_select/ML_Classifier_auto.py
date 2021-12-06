import pandas as pd
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.svm import SVC as svc
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def AUC(true, pred):
    if np.sum(true)==len(true) or np.sum(true)==0:
        return "ALL num are same"
    else:
        return auc(true, pred)

class cross_val():
    def __init__(self, model, X, y, random_state, cv):
        super(cross_val, self).__init__()

        if random_state != 0:
            index = np.arange(0, len(y))
            np.random.seed(random_state)
            np.random.shuffle(index)
            np.random.seed(random_state)
            np.random.shuffle(index)
        else:
            index = np.arange(len(y))

        step = int(len(y) / cv)
        self.acc = []
        self.f1 = []
        self.pred_all = np.zeros((len(y)), dtype=float)
        self.obs_all = np.zeros((len(y)), dtype=float)
        for i in range(cv):
            if i < cv - 1:
                index_train = np.concatenate([index[:i * step], index[(i + 1) * step:]], axis=0)
                index_val = index[i * step:(i + 1) * step]
            else:
                index_train = index[0:i * step]
                index_val = index[i * step:]

            X_train = X[index_train]
            y_train = y[index_train]
            X_val = X[index_val]
            y_val = y[index_val]

            pred = model.fit(X_train, y_train).predict(X_val)
            #             self.pred_all = np.concatenate([self.pred_all,pred], axis=0)
            self.pred_all[index_val] = pred
            self.obs_all[index_val] = y_val
            self.acc.append(accuracy_score(y_val, pred))
            self.f1.append(f1_score(y_val, pred))

        self.acc_mean = accuracy_score(self.obs_all, self.pred_all)
        self.f1_mean = f1_score(self.obs_all, self.pred_all)

def result_figure(y_test_train, pred_train, y_test_test, pred_test, data_neme, model_name, path):

    os.makedirs("{}/csv".format(path), exist_ok=True)

    acc_train = accuracy_score(y_test_train, pred_train)
    acc_test = accuracy_score(y_test_test, pred_test)

    f1_train = f1_score(y_test_train, pred_train)
    f1_test = f1_score(y_test_test, pred_test)

    auc_train = AUC(y_test_train, pred_train)
    auc_test = AUC(y_test_test, pred_test)

    pd.concat([pd.DataFrame(pred_train, columns=["prediction"])
                  , pd.DataFrame(y_test_train, columns=['true'])
                  , pd.DataFrame([acc_train], columns=["ACC"])
                  , pd.DataFrame([f1_train], columns=['f1_score'])
                  , pd.DataFrame([auc_train], columns=['AUC'])], axis=1).to_csv(
        "{}/csv/{}_{}_kf.csv".format(path, data_neme, model_name), index=None)

    pd.concat([pd.DataFrame(pred_test, columns=["prediction"])
                  , pd.DataFrame(y_test_test, columns=['true'])
                  , pd.DataFrame([acc_test], columns=["ACC"])
                  , pd.DataFrame([f1_test], columns=['f1_score'])
                  , pd.DataFrame([auc_test], columns=['AUC'])], axis=1).to_csv(
        "{}/csv/{}_{}_test.csv".format(path, data_neme, model_name), index=None)

    print("ACC-5cv = {:.3f} ACC-val = {:.3f}".format(acc_train, acc_test))
    print("f1_score-5cv = {:.3f} f1_score-val = {:.3f}".format(f1_train, f1_test))
    print("AUC-5cv = {} AUC-val = {}".format(auc_train, auc_test))

def rfc_auto(X, y, test_X, test_y, random_state, cv, data_name, model_name):
    print("*" * 100)
    print(data_name, f"Model:{model_name}")

    model = rfc(n_jobs=-1)
    before_kf_acc = cross_val(model, X, y, random_state, cv).acc_mean
    before_kf_f1 = cross_val(model, X, y, random_state, cv).f1_mean

    test_pred = model.fit(X, y).predict(test_X)
    before_test_acc = accuracy_score(test_y, test_pred)
    before_test_f1 = f1_score(test_y, test_pred)

    print("before kf ACC", before_kf_acc)
    print("before kf f1_score", before_kf_f1)
    print("before test ACC", before_test_acc)
    print("before test f1_score", before_test_f1)

    # 随机种子
    scores = []
    for i in range(0, 200):
        model = rfc(n_jobs=-1
                    , random_state=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(range(0, 200), scores)
    plt.show()

    random_state_rf = range(0, 200)[scores.index(max(scores))]
    print("random_state:", max(scores), random_state_rf)

    # 随机树数目
    scores = []
    for i in range(1, 200):
        model = rfc(n_estimators=i
                    , random_state=random_state_rf
                    , n_jobs=-1)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(range(1, 200), scores)
    plt.show()

    n_estimators = range(1, 200)[scores.index(max(scores))]
    print("n_estimators:", max(scores), n_estimators)

    # 最大深度
    scores = []
    for i in range(1, 200):
        model = rfc(n_estimators=n_estimators
                    , random_state=random_state_rf
                    , n_jobs=-1
                    , max_depth=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(range(1, 200), scores)
    plt.show()

    max_depth = range(1, 200)[scores.index(max(scores))]
    print("max_depth", max(scores), max_depth)

    # 最大特征选择
    scores = []
    for i in range(int(X.shape[1] ** 0.5), X.shape[1]):
        model = rfc(n_jobs=-1
                    , n_estimators=n_estimators
                    , random_state=random_state_rf
                    , max_depth=max_depth
                    , max_features=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(range(int(X.shape[1] ** 0.5), X.shape[1]), scores)
    plt.show()

    max_features = range(int(X.shape[1] ** 0.5), X.shape[1])[scores.index(max(scores))]
    print("max_features", max(scores), max_features)

    # 最小纯度递减
    scores = []
    for i in np.linspace(0, 0.5, 20):
        model = rfc(n_jobs=-1
                    , n_estimators=n_estimators
                    , random_state=random_state_rf
                    , max_depth=max_depth
                    , max_features=max_features
                    , min_impurity_decrease=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(np.linspace(0, 0.5, 20), scores)
    plt.show()

    min_impurity_decrease = np.linspace(0, 0.5, 20)[scores.index(max(scores))]
    print("min_impurity_decrease:", max(scores), min_impurity_decrease)

    # 最大样本数量
    scores = []
    len_train = int(len(y) / cv) * (cv - 1) + 1
    for i in range(1, len_train):
        model = rfc(n_jobs=-1
                    , n_estimators=n_estimators
                    , random_state=random_state_rf
                    , max_depth=max_depth
                    , max_features=max_features
                    , min_impurity_decrease=min_impurity_decrease
                    , max_samples=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(range(1, len_train), scores)
    plt.show()

    max_samples = range(1, len_train)[scores.index(max(scores))]
    print("max samples", max(scores), max_samples)

    ####################################

    model = rfc(n_jobs=-1
                , n_estimators=n_estimators
                , random_state=random_state_rf
                , max_depth=max_depth
                , max_features=max_features
                , min_impurity_decrease=min_impurity_decrease
                , max_samples=max_samples)
    score = cross_val(model, X, y, random_state, cv)
    after_kf_acc = score.acc_mean
    after_kf_f1 = score.f1_mean

    # 重新根据max_sample的大小范围等比例放缩
    max_samples = round(max_samples / len_train * len(y))
    model = rfc(n_jobs=-1
                , n_estimators=n_estimators
                , random_state=random_state_rf
                , max_depth=max_depth
                , max_features=max_features
                , min_impurity_decrease=min_impurity_decrease
                , max_samples=max_samples)
    test_pred = model.fit(X, y).predict(test_X)
    after_test_acc = accuracy_score(test_y, test_pred)
    after_test_f1 = f1_score(test_y, test_pred)

    print("kf ACC:", before_kf_acc, "->", after_kf_acc)
    print("kf f1_score:", before_kf_f1, "->", after_kf_f1)

    print("test ACC:", before_test_acc, "->", after_test_acc)
    print("test f1_score:", before_test_f1, "->", after_test_f1)

    print("n_estimators=", n_estimators, ","
          , "random_state=", random_state_rf, ","
          , "max_depth=", max_depth, ","
          , "max_features=", max_features, ","
          , "min_impurity_decrease=", min_impurity_decrease, ","
          , "max_samples=", max_samples)

    result_figure(y, score.pred_all, test_y, test_pred, data_name, model_name, "./results")
    return model, score.pred_all, test_pred

def svc_auto(X, y, test_X, test_y, random_state, cv, data_name, model_name):
    print("*" * 100)
    print(data_name, f"Model:{model_name}")

    model = svc()
    before_kf_acc = cross_val(model, X, y, 1, cv).acc_mean
    before_kf_f1 = cross_val(model, X, y, 1, cv).f1_mean

    test_pred = model.fit(X, y).predict(test_X)
    before_test_acc = accuracy_score(test_y, test_pred)
    before_test_f1 = f1_score(test_y, test_pred)

    print("before kf ACC", before_kf_acc)
    print("before kf f1_score", before_kf_f1)
    print("before test ACC", before_test_acc)
    print("before test f1_score", before_test_f1)

    # kernel
    scores = []
    for i in ["rbf", "poly", "sigmoid"]:
        model = svc(kernel=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.bar(["rbf", "poly", "sigmoid"], scores)
    plt.xticks(ticks=["rbf", "poly", "sigmoid"])
    plt.show()

    kernel = ["rbf", "poly", "sigmoid"][scores.index(max(scores))]
    print(max(scores), kernel)

    # C
    scores = []
    range_list = np.arange(1, 50, 0.05)
    for i in range_list:
        model = svc(kernel="rbf", C=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)

    plt.figure()
    plt.plot(range_list, scores)
    plt.show()

    C = range_list[scores.index(max(scores))]
    print(max(scores), C)

    # gamma
    scores = []
    range_list = np.linspace(0.0001, 100 / X.shape[1], 200)
    for i in range_list:
        model = svc(kernel=kernel, gamma=i, C=C)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)

    plt.figure()
    plt.plot(range_list, scores)
    plt.show()

    gamma = range_list[scores.index(max(scores))]
    if score.acc_mean > cross_val(svc(kernel=kernel, gamma="scale", C=C), X, y, random_state, cv).acc_mean:
        gamma = gamma
    else:
        gamma = 'scale'
    print(max(scores), gamma)

    ####################################
    model = svc(kernel=kernel, gamma=gamma, C=C)
    score = cross_val(model, X, y, 1, cv)
    after_kf_acc = score.acc_mean
    after_kf_f1 = score.f1_mean

    test_pred = model.fit(X, y).predict(test_X)
    after_test_acc = accuracy_score(test_y, test_pred)
    after_test_f1 = f1_score(test_y, test_pred)

    print("kf ACC:", before_kf_acc, "->", after_kf_acc)
    print("kf f1_score:", before_kf_f1, "->", after_kf_f1)

    print("test ACC:", before_test_acc, "->", after_test_acc)
    print("test f1_score:", before_test_f1, "->", after_test_f1)

    print("kernel=", kernel, ","
          , "C=", C, ","
          , "gamma=", gamma, ",")

    result_figure(y, score.pred_all, test_y, test_pred, data_name, model_name, "./results")
    return model, score.pred_all, test_pred

def knnc_auto(X, y, test_X, test_y, random_state, cv, data_name, model_name):
    print("*" * 100)
    print(data_name, f"Model:{model_name}")

    model = knnc()
    before_kf_acc = cross_val(model, X, y, 1, cv).acc_mean
    before_kf_f1 = cross_val(model, X, y, 1, cv).f1_mean

    test_pred = model.fit(X, y).predict(test_X)
    before_test_acc = accuracy_score(test_y, test_pred)
    before_test_f1 = f1_score(test_y, test_pred)

    print("before kf ACC", before_kf_acc)
    print("before kf f1_score", before_kf_f1)
    print("before test ACC", before_test_acc)
    print("before test f1_score", before_test_f1)

    # algorithm
    scores = []
    for i in ['ball_tree', 'kd_tree', 'brute']:
        model = knnc(n_jobs=-1, algorithm=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)

    plt.figure()
    plt.bar(['ball_tree', 'kd_tree', 'brute'], scores)
    plt.show()

    algorithm = ['ball_tree', 'kd_tree', 'brute'][scores.index(max(scores))]
    print(max(scores), algorithm)

    # weights
    scores = []
    for i in ['uniform', 'distance']:
        model = knnc(n_jobs=-1, algorithm=algorithm, weights=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)

    plt.figure()
    plt.bar(['uniform', 'distance'], scores)
    plt.show()

    weights = ['uniform', 'distance'][scores.index(max(scores))]
    print(max(scores), weights)

    # n_neighbors
    scores = []
    for i in range(1, 50):
        model = knnc(n_jobs=-1, algorithm=algorithm, weights=weights, n_neighbors=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)

    plt.figure()
    plt.plot(range(1, 50), scores)
    plt.show()

    n_neighbors = range(1, 50)[scores.index(max(scores))]
    print(max(scores), n_neighbors)

    # p
    scores = []
    for i in [1, 2]:
        model = knnc(n_jobs=-1, algorithm=algorithm, weights=weights, n_neighbors=n_neighbors, p=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)

    plt.figure()
    plt.bar([1, 2], scores)
    plt.show()

    p = [1, 2][scores.index(max(scores))]
    print(max(scores), p)

    # leaf_size
    scores = []
    for i in range(1, 100):
        model = knnc(n_jobs=-1, algorithm=algorithm, weights=weights, n_neighbors=n_neighbors, p=p, leaf_size=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)

    plt.figure()
    plt.plot(range(1, 100), scores)
    plt.show()

    leaf_size = range(1, 100)[scores.index(max(scores))]
    print(max(scores), leaf_size)

    ####################################
    model = knnc(n_jobs=-1, algorithm=algorithm, weights=weights, n_neighbors=n_neighbors, p=p, leaf_size=leaf_size)
    score = cross_val(model, X, y, 1, cv)
    after_kf_acc = score.acc_mean
    after_kf_f1 = score.f1_mean

    test_pred = model.fit(X, y).predict(test_X)
    after_test_acc = accuracy_score(test_y, test_pred)
    after_test_f1 = f1_score(test_y, test_pred)

    print("kf ACC:", before_kf_acc, "->", after_kf_acc)
    print("kf f1_score:", before_kf_f1, "->", after_kf_f1)

    print("test ACC:", before_test_acc, "->", after_test_acc)
    print("test f1_score:", before_test_f1, "->", after_test_f1)

    print("algorithm=", algorithm, ","
          , "weights=", weights, ","
          , "n_neighbors=", n_neighbors, ","
          , "p=", p, ","
          , "leaf_size=", leaf_size)

    result_figure(y, score.pred_all, test_y, test_pred, data_name, model_name, "./results")
    return model, score.pred_all, test_pred

def xgbc_auto(X, y, test_X, test_y, random_state, cv, data_name, model_name):
    print("*" * 100)
    print(data_name, f"Model:{model_name}")
    model = XGBClassifier(eval_metric=['logloss','auc','error'], use_label_encoder=False, n_jobs=-1
                          , objective='binary:logistic')
    before_kf_acc = cross_val(model, X, y, random_state, cv).acc_mean
    before_kf_f1 = cross_val(model, X, y, random_state, cv).f1_mean

    test_pred = model.fit(X, y).predict(test_X)
    before_test_acc = accuracy_score(test_y, test_pred)
    before_test_f1 = f1_score(test_y, test_pred)

    print("before kf ACC", before_kf_acc)
    print("before kf f1_score", before_kf_f1)
    print("before test ACC", before_test_acc)
    print("before test f1_score", before_test_f1)

    # num_boost and eta
    scores = np.zeros((len(range(1, 101)), len(np.arange(0.01, 0.5, 0.01))), dtype=np.float64)
    score = 0
    imax = 0
    jmax = 0
    for i, n_estimators in enumerate(range(1, 101)):
        for j, eta in enumerate(np.arange(0.01, 0.5, 0.01)):
            model = XGBClassifier(eval_metric=['logloss','auc','error'], use_label_encoder=False, n_jobs=-1
                                  , objective='binary:logistic', eta=eta, n_estimators=n_estimators)
            scores[i][j] = cross_val(model, X, y, random_state, cv).acc_mean

            if scores[i][j] > score:
                score = scores[i][j]
                jmax = j
                imax = i

    fig = plt.figure(figsize=[10, 10])
    ax3d = Axes3D(fig)
    X_ax, y_ax = np.meshgrid(range(1, 101), np.arange(0.01, 0.5, 0.01))
    matrix = np.array(scores.T)
    ax3d.plot_surface(X_ax, y_ax, matrix, linewidth=0, antialiased=False, shade=True, alpha=0.5,
                      cmap='rainbow')
    plt.show()

    n_estimators = range(1, 101)[imax]
    eta = np.arange(0.01, 0.5, 0.01)[jmax]
    print(score, "n_estimators:", n_estimators, "\n", "eta:", eta)

    # max_depth
    scores = []
    for i in range(1, 50):
        model = XGBClassifier(eval_metric=['logloss','auc','error'], use_label_encoder=False, n_jobs=-1
                              , objective='binary:logistic', eta=eta, n_estimators=n_estimators,
                              max_depth=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(range(1, 50), scores)
    plt.show()

    max_depth = range(1, 50)[scores.index(max(scores))]
    print("max_depth:", max(scores), max_depth)

    # gamma
    scores = []
    for i in np.arange(0, 5, 0.5):
        model = XGBClassifier(eval_metric=['logloss','auc','error'], use_label_encoder=False, n_jobs=-1
                              , objective='binary:logistic', eta=eta, n_estimators=n_estimators,
                              max_depth=max_depth, gamma=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(np.arange(0, 5, 0.5), scores)
    plt.show()

    gamma = np.arange(0, 5, 0.5)[scores.index(max(scores))]
    print("gamma", max(scores), gamma)

    # reg_alpha
    scores = []
    for i in np.arange(0, 5, 0.05):
        model = XGBClassifier(eval_metric=['logloss','auc','error'], use_label_encoder=False, n_jobs=-1
                              , objective='binary:logistic', eta=eta, n_estimators=n_estimators,
                              max_depth=max_depth, gamma=gamma, reg_alpha=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(np.arange(0, 5, 0.05), scores)
    plt.show()

    reg_alpha = np.arange(0, 5, 0.05)[scores.index(max(scores))]
    print("reg_alpha", max(scores), reg_alpha)

    # reg_lambda
    scores = []
    for i in np.arange(0, 5, 0.05):
        model = XGBClassifier(eval_metric=['logloss','auc','error'], use_label_encoder=False, n_jobs=-1
                              , objective='binary:logistic', eta=eta, n_estimators=n_estimators,
                              max_depth=max_depth, gamma=gamma, reg_alpha=reg_alpha, reg_lambda=i)
        score = cross_val(model, X, y, random_state, cv)
        scores.append(score.acc_mean)
        pass

    plt.figure()
    plt.plot(np.arange(0, 5, 0.05), scores)
    plt.show()

    reg_lambda = np.arange(0, 5, 0.05)[scores.index(max(scores))]
    print("reg_lambda:", max(scores), reg_lambda)

    ####################################

    model = XGBClassifier(eval_metric=['logloss','auc','error'], use_label_encoder=False, n_jobs=-1
                          , objective='binary:logistic', eta=eta, n_estimators=n_estimators,
                          max_depth=max_depth, gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    score = cross_val(model, X, y, random_state, cv)
    after_kf_acc = score.acc_mean
    after_kf_f1 = score.f1_mean

    test_pred = model.fit(X, y).predict(test_X)
    after_test_acc = accuracy_score(test_y, test_pred)
    after_test_f1 = f1_score(test_y, test_pred)

    print("kf ACC:", before_kf_acc, "->", after_kf_acc)
    print("kf f1_score:", before_kf_f1, "->", after_kf_f1)

    print("test ACC:", before_test_acc, "->", after_test_acc)
    print("test f1_score:", before_test_f1, "->", after_test_f1)

    print('objective', 'binary:logistic'
          ,'n_estimators', n_estimators
          , 'eta', eta
          , 'max_depth', max_depth
          , 'gamma', gamma
          , 'reg_alpha', reg_alpha
          , 'reg_lambda', reg_lambda)

    result_figure(y, score.pred_all, test_y, test_pred, data_name, model_name, "./results")
    return model, score.pred_all, test_pred
import pandas as pd
import numpy as np
import copy
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from torch.optim import lr_scheduler

batch_size = 16

class Net_FCNN_c(Module):
    def __init__(self, in_size):
        super(Net_FCNN_c, self).__init__()
        self.fc1 = nn.Linear(in_size, 2000)
        self.fc2 = nn.Linear(2000, 4000)
        self.fc3 = nn.Linear(4000, 200)
        self.fc4 = nn.Linear(200, 2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(F.relu(self.fc3(x)))
        x = self.fc4(x)

        return x

def train_Net_c(trainloader, testloader, epochs, model_ori):
    import copy
    model = copy.deepcopy(model_ori)
    # model.cuda()
    best_r2 = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)  # , momentum=0.5)
    train_mse = []
    test_mse = []
    test_r2 = []
    loss_all = 0
    for epoch in range(epochs):
        batch_loss = []
        for inputs, target in trainloader:
            #             inputs = inputs.cuda()
            #             target = target.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            #             print(outputs.shape)
            #             print(target.shape)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        batch_loss = np.vstack(batch_loss).mean()
        train_mse.append(batch_loss)
        print("test", "epoch:", epoch, batch_loss)

        pred = []
        obs = []
        for inputs, target in testloader:
            #             inputs = inputs.cuda()
            #             target = target.cuda()
            outputs = model(inputs)

            ## 与ANN不一样的地方
            _, predicted = torch.max(outputs.data, dim=1)
            [pred.append(i) for i in predicted]
            [obs.append(i) for i in target.data]

        #         pred = np.vstack(pred).ravel()
        #         obs = np.vstack(obs).ravel()
        r2 = accuracy_score(obs, pred)
        Mse = f1_score(obs, pred)
        test_mse.append(Mse)
        test_r2.append(r2)
        if best_r2 < r2:
            best_pred = pred
            best_r2 = r2
            best_mse = Mse
        print('r2 on test: %.3f %%' % (100 * r2), 'mse on test: %.5f \n' % (Mse))

    return best_mse, best_r2, best_pred, obs

def T_f(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype)

class cross_val_Classifier():
    def __init__(self, model, epochs, X_y, random_state, cv):
        if random_state != 0:
            index = np.arange(0, len(X_y))
            np.random.seed(random_state)
            np.random.shuffle(index)
            np.random.seed(random_state)
            np.random.shuffle(index)
        else:
            index = np.arange(len(X_y))

        step = int(len(X_y) / cv)
        self.pred_all = np.zeros((len(X_y)), dtype=float)
        self.obs_all = np.zeros((len(X_y)), dtype=float)
        for i in range(cv):
            if i < cv - 1:
                index_train = np.concatenate([index[:i * step], index[(i + 1) * step:]], axis=0)
                index_val = index[i * step:(i + 1) * step]
            else:
                index_train = index[0:i * step]
                index_val = index[i * step:]

            train_dataset = [X_y[i] for i in index_train]
            test_dataset = [X_y[i] for i in index_val]

            trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                      shuffle=False)  # ,num_workers=workers)
            testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

            _, _, self.pred_all[index_val], self.obs_all[index_val] = train_Net_c(trainloader, testloader, epochs,
                                                                                  model)

        self.r2_mean = accuracy_score(self.obs_all, self.pred_all)
        self.mse_mean = f1_score(self.obs_all, self.pred_all)

class Net_CNN_c(Module):
    def __init__(self):
        super(Net_CNN_c, self).__init__()
        self.Conv1 = nn.Conv1d(1, 10, kernel_size=5)  # 1000-996-498
        self.Conv2 = nn.Conv1d(10, 100, kernel_size=5)  # 498-494-247
        self.Conv3 = nn.Conv1d(100, 50, kernel_size=6)  # 247-242-121
        self.Conv4 = nn.Conv1d(50, 20, kernel_size=6)  # 121-116-58
        self.Conv5 = nn.Conv1d(20, 10, kernel_size=5)  # 58-54-27
        self.fc1 = nn.Linear(270, 50)
        self.fc2 = nn.Linear(50, 2)
        self.mp = nn.MaxPool1d(2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        #         x = self.bn(x)
        x = F.relu(x)
        x = F.relu(self.Conv1(x))
        x = self.drop(self.mp(x))
        x = F.relu(self.Conv2(x))
        x = self.drop(self.mp(x))
        x = F.relu(self.Conv3(x))
        x = self.drop(self.mp(x))
        x = F.relu(self.Conv4(x))
        x = self.drop(self.mp(x))
        x = F.relu(self.Conv5(x))
        x = self.drop(self.mp(x))

        x = x.view(-1, 270)  ##1760
        x = self.fc1(x)
        x = self.drop(x)

        x = self.fc2(x)

        return x


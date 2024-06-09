import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, precision_score, f1_score, roc_curve, auc
from fine_s import SCNN
from Model.Student import SCNN as CNN
from Model.Teacher import TCNN
import wandb
from torch import nn
from torch.utils.data import DataLoader
from utils.DatasetLoader import CustomTensorDataset
from utils.Preprocess import prepro
from utils.loss_fun import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
print(torch.cuda.is_available())

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def inference(dataloader, model):
    net = model
    y_list, y_predict_list, y_prob_list = [], [], []
    if use_gpu:
        net.cuda()
    net.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.type(torch.float)
            y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            y_prob = torch.softmax(y_hat, dim=1)
            y_predict = y_hat.argmax(dim=1)
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())
            y_prob_list.extend(y_prob.detach().cpu().numpy())

        cnf_matrix = confusion_matrix(y_list, y_predict_list)
        recall = recall_score(y_list, y_predict_list, average="macro")
        precision = precision_score(y_list, y_predict_list, average="macro")
        F1 = f1_score(y_list, y_predict_list, average="macro")
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        TN = TN.astype(float)
        FPR = np.mean(FP / (FP + TN))
        print(F1, FPR, recall, precision)

        return F1, FPR, recall, precision, y_list, y_predict_list, y_prob_list

def inference10(net0):
    test_f1 = []
    test_FPR = []
    test_recall = []
    test_precision = []
    seed = 43
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for seed in range(seed, seed+3):
        random_seed(seed)

        train_X, train_Y, valid_X, valid_Y = prepro(d_path='data/0HP',
                                                    length=2048,
                                                    number=750,
                                                    normal=True,
                                                    enc=True,
                                                    enc_step=28,
                                                    snr=-6,
                                                    property='Train',
                                                    noise=True)

        test_X, test_Y = prepro(d_path='data/0HP',
                                length=2048,
                                number=250,
                                normal=True,
                                enc=True,
                                enc_step=28,
                                snr=-6,
                                property='Test',
                                noise=True)

        train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]

        test_dataset = CustomTensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_Y))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        f1, FPR, recall, precision, y_list, y_predict_list, y_prob_list = inference(test_loader, net0)
        print(seed)
        test_f1.append(f1)
        test_FPR.append(FPR)
        test_recall.append(recall)
        test_precision.append(precision)
        all_y_true.extend(y_list)
        all_y_pred.extend(y_predict_list)
        all_y_prob.extend(y_prob_list)

    print("test_f1: mean: ", np.mean(test_f1), "var: ", np.var(test_f1))
    print("test_FPR: mean: ", np.mean(test_FPR), "var: ", np.var(test_FPR))
    print("test_recall: mean: ", np.mean(test_recall), "var: ", np.var(test_recall))
    print("test_precision: mean: ", np.mean(test_precision), "var: ", np.var(test_precision))

    return all_y_true, all_y_pred, all_y_prob


import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    # file_t = "Pth\DKD11_Tmodel-6_0HP_1.0.pth"
    # tcnn = TCNN()
    # ckpt_t = torch.load(file_t, map_location=lambda storage, loc: storage.cuda())
    # tcnn.load_state_dict(ckpt_t)
    # print(tcnn)
    # y_true, y_pred, y_prob = inference10(tcnn)

    # file_s = "Pth\DKD11_Smodel-6_0HP_0.99.pth"
    # s_cnn = CNN()
    # ckpt_s = torch.load(file_s, map_location=lambda storage, loc: storage.cuda())
    # s_cnn.load_state_dict(ckpt_s)
    # print(s_cnn)
    # y_true, y_pred, y_prob = inference10(s_cnn)


    # print("加载模型")
    # file = "Pth/LoRA_Best_model_FC3.pth"
    # ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    # scnn = SCNN(12)
    # scnn.load_state_dict(ckpt)
    # print(scnn)
    # y_true, y_pred, y_prob = inference10(scnn)

    # # Generate confusion matrix
    # cnf_matrix = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

    # # Compute ROC curve and AUC for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # n_classes = len(np.unique(y_true))
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(np.array(y_true) == i, np.array(y_prob)[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # print(roc_auc)
    # # Plot all ROC curves
    # plt.figure(figsize=(8, 6))
    # for i in range(n_classes):
    #     plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.6f})'.format(i, roc_auc[i]))

    # # Plot a random guess line
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing')

    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (Recall)')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.show()
    file_s = "Pth/DKD11_Smodel-6_0HP_0.99.pth"
    s_cnn = CNN()
    ckpt_s = torch.load(file_s, map_location=lambda storage, loc: storage.cuda())
    s_cnn.load_state_dict(ckpt_s)
    print(s_cnn)
    y_true_s, y_pred_s, y_prob_s = inference10(s_cnn)

    print("加载模型")
    file = "Pth/LoRA_Best_model_FC3.pth"
    ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    scnn = SCNN(12)
    scnn.load_state_dict(ckpt)
    print(scnn)
    y_true, y_pred, y_prob = inference10(scnn)

    # 计算第一个模型的ROC曲线
    fpr_s, tpr_s, _ = roc_curve(y_true_s, y_prob_s)
    roc_auc_s = auc(fpr_s, tpr_s)

    # 计算第二个模型的ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # 绘制两个模型的ROC曲线
    plt.figure()
    plt.plot(fpr_s, tpr_s, color='blue', lw=2, label='Model 1 (AUC = %0.2f)' % roc_auc_s)
    plt.plot(fpr, tpr, color='red', lw=2, label='Model 2 (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
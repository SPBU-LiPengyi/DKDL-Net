'''
Test Model
'''
import os
import random
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, \
    precision_score, f1_score
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
    y_list, y_predict_list = [], []
    if use_gpu:
        net.cuda()
    net.eval()
    # endregion
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.type(torch.float)
            y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            y_predict = y_hat.argmax(dim=1)
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())
            


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

        return F1,FPR,recall,precision


def inference10(net0):
    test_f1 = []
    test_FPR = []
    test_recall = []
    test_precision = []
    seed = 45
    for seed in range(seed, seed + 1):
        random_seed(seed)

        train_X, train_Y, valid_X, valid_Y = prepro(d_path='data/0HP',
                                                    length=2048,
                                                    number=750,
                                                    normal=True,
                                                    enc=True,
                                                    enc_step=28,
                                                    snr=-6,
                                                    property='Train',
                                                    noise=True
                                                    )

        # test set, number denotes each category has 250 samples
        test_X, test_Y = prepro(d_path='data/0HP',
                                length=2048,
                                number=250,
                                normal=True,
                                enc=True,
                                enc_step=28,
                                snr=-6,
                                property='Test',
                                noise=True
                                )
        train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]

        test_dataset = CustomTensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_Y))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        f1, FPR, recall, precision = inference(test_loader, net0)
        test_f1.append(f1)
        test_FPR.append(FPR)
        test_recall.append(recall)
        test_precision.append(precision)

    print("test_f1: mean: ", np.mean(test_f1), "var: ", np.var(test_f1))
    print("test_FPR: mean: ", np.mean(test_FPR), "var: ", np.var(test_FPR))
    print("test_recall: mean: ", np.mean(test_recall), "var: ", np.var(test_recall))
    print("test_precision: mean: ", np.mean(test_precision), "var: ", np.var(test_precision))


if __name__ == "__main__":
    
    # LoRA Student
    print("加载LoRA学生模型")
    file = "Pth\LoRA_Best_model_FC3.pth"
    ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    scnn = SCNN(12)
    scnn.load_state_dict(ckpt)
    print(scnn)
    start = time.time()
    inference10(scnn)
    stop = time.time()
    print(stop-start)
    print((stop-start)/2500)

    # # Student 
    # print("加载学生模型")
    # cnn = CNN()
    # file = "Pth\DKD11_Smodel-6_0HP_0.99.pth"
    # ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    # cnn.load_state_dict(ckpt)
    # print(cnn)
    # start = time.time()
    # inference10(cnn)
    # stop = time.time()
    # print(stop-start)
    # print((stop-start)/2500)

    # Teacher 
    print("加载LoRA学生模型")
    tcnn = TCNN()
    file = "Pth\DKD11_Tmodel-6_0HP_1.0.pth"
    ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    tcnn.load_state_dict(ckpt)
    print(tcnn)
    start = time.time()
    inference10(tcnn)
    stop = time.time()
    print(stop-start)
    print((stop-start)/2500)
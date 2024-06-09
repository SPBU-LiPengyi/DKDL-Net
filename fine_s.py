import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Preprocess import prepro
from utils.DatasetLoader import CustomTensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis, flop_count_str


class LoRAConv1d(nn.Module):
    def __init__(self, original_layer, r):
        super(LoRAConv1d, self).__init__()
        self.original_layer = original_layer
        self.rank = r
        self.lora_A = nn.Parameter(torch.randn(original_layer.out_channels, r))
        self.lora_B = nn.Parameter(torch.randn(r, original_layer.in_channels * original_layer.kernel_size[0]))

    def forward(self, x):
        out = self.original_layer(x)
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(
            self.original_layer.out_channels, self.original_layer.in_channels, self.original_layer.kernel_size[0]
        )
        out += nn.functional.conv1d(x, lora_update, stride=self.original_layer.stride, padding=self.original_layer.padding)
        return out

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r):
        super(LoRALinear, self).__init__()
        self.original_layer = original_layer
        self.rank = r
        self.lora_A = nn.Parameter(torch.randn(original_layer.out_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, original_layer.in_features))

    def forward(self, x):
        out = self.original_layer(x)
        lora_update = torch.matmul(self.lora_A, self.lora_B)
        out += nn.functional.linear(x, lora_update)
        return out

class SCNN(nn.Module):
    def __init__(self, lora_rank=4):
        super(SCNN, self).__init__()
        self.conv1 = LoRAConv1d(nn.Conv1d(in_channels=1, out_channels=4, kernel_size=64, stride=8, padding=28), lora_rank)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = LoRALinear(nn.Linear(in_features=256, out_features=10), lora_rank)

    def forward(self, x):
        x = abs(torch.fft.fft(x, dim=2, norm="forward"))
        _, x = x.chunk(2, dim=2)
        x1 = self.conv1(x)
        x2 = nn.functional.relu(x1)
        x3 = self.pool1(x2)
        x4 = x3.view(-1, 256)
        x5 = self.fc1(x4)
        return x5

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # TCNN = SCNN()
    # print(TCNN)
    # file = "Pth\LoRA_Best_model_FC31.0.pth"
    # ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    # total_parameter = 0
    # for name, param in TCNN.state_dict(ckpt).items():
    #     total_parameter = sum(param.numel() for param in TCNN.parameters())
    #     print(f"{name}: {param.shape}")
    # print(total_parameter)

    # 训练随机性

    random_seed(42)

    TCNN = SCNN(12)
    file = file = "Pth\DKD11_Smodel-6_0HP_0.99.pth"
    ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    total_parameter = 0
    
    # 读取模型参数
    for name, param in TCNN.state_dict(ckpt).items():
        total_parameter = sum(param.numel() for param in TCNN.parameters())
        print(f"{name}: {param.shape}")


        # print('='*20)
    
    print(TCNN)
    
    print(total_parameter)

    # 基于LoRA微调模型训练
    # 加载数据
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
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
    train_X, valid_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :]
    
    train_dataset = CustomTensorDataset(torch.tensor(train_X, dtype=torch.float), torch.tensor(train_Y))
    valid_dataset = CustomTensorDataset(torch.tensor(valid_X, dtype=torch.float), torch.tensor(valid_Y))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, drop_last=True)
    
    data_loaders = {
        "train": train_loader,
        "validation": valid_loader
    }
    
    epochs = 70
    # 定义训练模型
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    use_gpu = torch.cuda.is_available()
    
    TCNN.cuda()
    
    acc_max = 0
    
    loss_func = nn.CrossEntropyLoss()
    for e in range(epochs):
        loss = 0
        total = 0
        correct = 0
        correct_v = 0
        total_v = 0
        loss_total = 0
        
        for step, (x, y) in enumerate(data_loaders['train']):
            TCNN.train()
            x = x.type(torch.float).cuda()
            y = y.type(torch.long)
            y = y.view(-1).cuda()
    
            # optimizer = torch.optim.SGD(TCNN.parameters(), lr=0.01, momentum=0.9)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
            #                                                            eta_min=1e-8)

            optimizer = torch.optim.Adam(TCNN.parameters(), lr=0.005, weight_decay=0.0001)
                
            y_t = TCNN(x).cuda()

            hard_loss = loss_func(y_t, y)
            
            optimizer.zero_grad()
            hard_loss.backward()
            optimizer.step()

            loss_total += hard_loss.item()
            y_predict = y_t.argmax(dim=1)

            total += y.size(0)
            correct += (y_predict == y).cpu().squeeze().sum().numpy()
            
            if step % 20 == 0:
                print('Epoch:%d, Step [%d/%d], Loss: %.4f'% (e + 1, step + 1, len(data_loaders['train']), loss_total / len(data_loaders['train'].dataset)))
                
            loss_total = loss_total / len(data_loaders['train'].dataset)
            acc = correct / total
            train_loss.append(loss_total)
            train_acc.append(acc)

        for step, (x, y) in enumerate(data_loaders["validation"]):
            TCNN.eval()
            torch.no_grad()
            x = x.type(torch.float).cuda()
            y = y.type(torch.long)
            y = y.view(-1).cuda()
            y_t_v = TCNN(x).cuda()

            y_predict = y_t_v.argmax(dim=1)
            total_v += y.size(0)
            correct_v += (y_predict == y).cpu().squeeze().sum().numpy()
            if step % 20 == 0:
                print('Epoch:%d, Step [%d/%d], Loss: %.4f'% (e + 1, step + 1, len(data_loaders['validation']), loss_total / len(data_loaders['validation'].dataset)))
            acc_v = correct_v / total_v
            if acc_v >= acc_max:
                    acc_max = acc
                    # file = "Pth/" + config.stdut + "_best" + str(acc_max.item()) + ".pth"
                    # torch.save(net.state_dict(), file)
                    lora_best = TCNN
    file = "Pth/" + "LoRA_Best_model_FC3" + ".pth"
    torch.save(lora_best.state_dict(), file)

    ckpt_lora = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    
    print(acc)
    print("评估数据集：" + str(acc_max))
    print("Train End")
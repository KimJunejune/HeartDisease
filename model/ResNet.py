#-----------------------------
# 使用迁移学习ResNet进行预测。
# 同样，将原始counts分为0.25和0.75
#
# 样本1 - 7 为连续的7天的数据。 将第7天的数据作为目标值进行回归。
#-----------------------------


import os
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image

# torch
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 使用ACS做二维卷积
data_path = '../data'
count_path = os.path.join(
    data_path, 'counts_ACS.csv'
)
ACS_path = os.path.join(
    data_path, 'newset.csv'
)
# 只用当天的天气数据
df_features = pd.read_csv(ACS_path)
df_counts = pd.read_csv(count_path)
df_features['time'] = pd.to_datetime(df_features['time'])
df_counts['time'] = pd.to_datetime(df_counts['time'])
table = pd.merge(left=df_features, right=df_counts, on = 'time')
table['counts'] = table['counts'].replace(np.nan, table['counts'].median())
table['counts'] = table['counts'].replace(-1, table['counts'].median())
table = table.fillna(method='ffill', axis = 0)
table.to_csv(os.path.join(data_path, "mergedNewAcs.csv"), index=False)


#----------------------------------
# 阈值处理
# 根据画出的散点图进行clip操作
#----------------------------------
table['AQI_0'] = table['AQI_0'].clip(upper = 400)
table['AQI_6'] = table['AQI_6'].clip(upper = 400)
table['AQI_12'] = table['AQI_12'].clip(upper = 300)
table['AQI_18'] = table['AQI_18'].clip(upper = 400)

table['PM2_0'] = table['PM2_0'].clip(upper = 300)
table['PM2_6'] = table['PM2_6'].clip(upper = 300)
table['PM2_12'] = table['PM2_12'].clip(upper = 250)
table['PM2_18'] = table['PM2_18'].clip(upper = 300)

table['PM10_0'] = table['PM10_0'].clip(upper = 400)
table['PM10_6'] = table['PM10_6'].clip(upper = 400)
table['PM10_12'] = table['PM10_12'].clip(upper = 300)
table['PM10_18'] = table['PM10_18'].clip(upper = 400)

table['SO2_0'] = table['SO2_0'].clip(upper = 100)
table['SO2_6'] = table['SO2_6'].clip(upper = 100)
table['SO2_12'] = table['SO2_12'].clip(upper = 100)
table['SO2_18'] = table['SO2_18'].clip(upper = 100)

table['NO2_0'] = table['NO2_0'].clip(upper = 120)
table['NO2_6'] = table['NO2_6'].clip(upper = 120)
table['NO2_12'] = table['NO2_12'].clip(upper = 120)
table['NO2_18'] = table['NO2_18'].clip(upper = 120)

table['CO_0'] = table['CO_0'].clip(upper = 4)
table['CO_6'] = table['CO_6'].clip(upper = 4)
table['CO_12'] = table['CO_12'].clip(upper = 4)
table['CO_18'] = table['CO_18'].clip(upper = 4)


table['pressure_0'] = table['pressure_0'].clip(lower = 9950)
table['pressure_6'] = table['pressure_6'].clip(lower = 9950)
table['pressure_12'] = table['pressure_12'].clip(lower = 9950)
table['pressure_18'] = table['pressure_18'].clip(lower = 9950)

table.iloc[728:732, 1:29] = np.nan
table = table.fillna(table.mean())

for row in table.columns[1:-1]:
    table.loc[:, row] = (table.loc[:, row] - table.loc[:, row].mean()) / table.loc[:, row].std()

# ---------------------------------
# 在notebook中发现 四分位数是 5， 8， 11
# ---------------------------------
def filter(x):
    if x<=5:
        return 0
    elif x>=11:
        return 1
    else:
        return 2
table['counts'] = table.counts.apply(lambda x:filter(x))
table = table[(table['counts'] == 0) | (table['counts'] == 1)]

# ----------------------------------
# 2016/12/25 -- 2016-12/31 的标签记为 2017/01/01的counts
# 2016/12/26 -- 2017-01/01 的标签记为 2017/01/02的counts
# 以此类推
# ----------------------------------
def to_img(df):

    imgs = []
    labels = []
    for col in range(df.shape[0] - 7):
        temp_df = df.iloc[col:col+8]
        img = temp_df.iloc[0:7, 1:-1]
        img = img.values
        lable = temp_df.iloc[7, -1]
        imgs.append(img)
        labels.append(lable)
    return imgs, labels

# 分割数据集
train_ratio = 0.7
fir, sec = to_img(table)
data =[[], []]
data[0].extend(fir[:1096])
data[0].extend(fir[1338:])
data[1].extend(sec[:1096])
data[1].extend(sec[1338:])

datasets = {
    'train': [[], []],
    'val': [[], []]
}

for i in range(len(data[0])):
    if random.uniform(0, 1) < train_ratio:
        datasets['train'][0].append(data[0][i])
        datasets['train'][1].append(data[1][i])
    else:
        datasets['val'][0].append(data[0][i])
        datasets['val'][1].append(data[1][i])

# -------------------------
# 自定义Dataset和DataLoader
# 使用PIL.Image()将由天气因子构成的假图片变为三通道， 实际操作为重叠数据
# -------------------------
class myData(Dataset):
    def __init__(self, datasets, mode):
        super(myData, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        self.imgs, self.labels = datasets[mode][0], datasets[mode][1]

    def __getitem__(self, idx):
        sample = Image.fromarray(self.imgs[idx])
        sample = sample.convert('RGB')
        sample = self.transform(sample)
        return sample, self.labels[idx]

    def __len__(self):
        return len(self.imgs)
# Dataset
img_datasets = {
    x: myData(datasets, x) for x in ['train', 'val']
}
dataset_sizes = {
    x: len(img_datasets[x]) for x in ['train', 'val']
}

# DataLoader
train_loader = DataLoader(
    img_datasets['train'],
    batch_size=1,
    shuffle=True,
)
val_loader = DataLoader(
    img_datasets['val'],
    batch_size=1,
    shuffle=False
)
dataloaders = {
    'train': train_loader,
    'val': val_loader
}

# ---------------------------
# 定义模型训练函数
# 如果在DataLoader中想使用num_workers 进行多线程操作的话
# 则必须包装训练过程，并在if __name__ == '__main__": 中调用
# ---------------------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, schedule, epoches=75):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epoches):
        print("in epoch: %d" % epoch)
        count = 0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # inputs = inputs.type(torch.FloatTensor).to("cuda:0")
                    # print(labels)
                    outputs = model(inputs)
                    # print(outputs)
                    _, idx = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(idx == labels.data)
            if phase == 'train':
                schedule.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

# -------------------------------------
# 迁移学习 使用Resnet18
# 采取第一种方法————全体参数微调
# 因为官网上的预训练模型并不一定适用于本项目
# 所以采取先全体微调，后冻结微调的策略
# -------------------------------------
model_res = models.resnet18(pretrained=True)
num_ftrs = model_res.fc.in_features
model_res.fc = nn.Linear(num_ftrs, 2)
model_res = model_res.to("cuda:0")

criterion = nn.CrossEntropyLoss()
optimizer_res = torch.optim.SGD(model_res.parameters(), lr = 0.02, momentum=0.1)
lr_sche_res = torch.optim.lr_scheduler.StepLR(optimizer_res, step_size=10, gamma=0.1)

model_res = train_model(model_res, criterion, optimizer_res, lr_sche_res, epoches=30)

# --------------------------------------
# 保存最佳模型参数，一遍下一次调用
# --------------------------------------
def vali(M ,dataset):
    M.eval()
    with torch.no_grad():
        correct = 0
        for (data, target) in val_loader:
            data, target = data.to("cuda:0"), target.to("cuda:0")

            pred = M(data)
            _, id = torch.max(pred, 1)
            correct += torch.sum(id == target.data)
        print("test accu: %.03f%%" % (100 * correct / len(dataset)))
    return (100 * correct / len(dataset)).item()
test_accu = int(vali(model_res, img_datasets['val']) * 100)
model_name = 'val_{}.pkl'.format(test_accu)

torch.save(model_res.state_dict(), os.path.join("../myModel", model_name))

# ---------------------------------------
# 调用从上一阶段保存的模型参数
# 冻结除全连接外的所有层
# 进行最后一层的更新
# ----------------------------------------
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load(os.path.join("../myModel", model_name)))
# 冻结所有层
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to("cuda:0")
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)    # 注意 仅更新全连接层
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,epoches=15)    # 模型训练

# --------------------------------------
# 最后保存最终结果
# --------------------------------------
test_accu = int(vali(model_ft, img_datasets['val']) * 100)
model_name = 'val_{}.pkl'.format(test_accu)

torch.save(model_ft.state_dict(), os.path.join("../myModel", model_name))
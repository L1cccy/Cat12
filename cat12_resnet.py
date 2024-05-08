import random
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models

# 分割训练集
train_ratio = 0.7

train_paths, train_labels = [], []
valid_paths, valid_labels = [], []
with open('E:\pycharm\PyCharm Community Edition\cat12\\train_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if random.uniform(0, 1) < train_ratio:
            train_paths.append(line.split('	')[0])
            label = line.split('	')[1]
            train_labels.append(int(line.split('	')[1]))
        else:
            valid_paths.append(line.split('	')[0])
            valid_labels.append(int(line.split('	')[1]))


class TrainData(Dataset):
    def __init__(self):
        super().__init__()
        self.color_jitter = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        self.normalize = T.Normalize(mean=0, std=1)
        self.random_crop = T.RandomCrop(256, pad_if_needed=True)
        self.random_rotation = T.RandomRotation(degrees=10)

    def __getitem__(self, idx):
        # 读取图片
        image_path = train_paths[idx]
        totensor = T.ToTensor()
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = totensor(image)

        # 图像增广
        features = self.color_jitter(image)
        features = self.random_rotation(features)
        features = self.random_crop(features)
        features = self.normalize(features)

        # 读取标签
        labels = train_labels[idx]

        return features, labels

    def __len__(self):
        return len(train_paths)


class ValidData(Dataset):
    def __init__(self):
        super().__init__()
        self.normalize = T.Normalize(mean=0, std=1)

    def __getitem__(self, idx):
        # 读取图片
        image_path = valid_paths[idx]
        totensor = T.ToTensor()
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = totensor(image)

        # 图像变换
        resize = T.Resize((256, 256))
        features = resize(image)
        features = self.normalize(features)

        # 读取标签
        labels = valid_labels[idx]

        return features, labels

    def __len__(self):
        return len(valid_paths)

# 直观验证增强数据可行性
train_data = TrainData()
# img, labels = train_data.__getitem__(98)
# img = torch.permute(img, [1, 2, 0])
# img_scaled = (img * 255).round().clamp(0, 255)
# img_uint8 = img_scaled.to(torch.uint8)
# plt.figure(dpi=40, figsize=(16,16))
# plt.imshow(img_uint8)
# plt.show()

valid_data = ValidData()
# img, label = valid_data.__getitem__(134)
# img = torch.permute(img, [1, 2, 0])
# img_scaled = (img * 255).round().clamp(0, 255)
# img_uint8 = img_scaled.to(torch.uint8)
# plt.figure(dpi=40,figsize=(16,16))
# plt.imshow(img_uint8)
# plt.show()

# dataloader
train_dataloader =DataLoader(train_data, batch_size=64, shuffle=True)
valid_dataloader =DataLoader(valid_data, batch_size=64, shuffle=True)

train_data_size = len(train_data)
valid_data_size = len(valid_data)

# 调用resnet并修改输出层神经元
model = models.resnet50()
in_features = model.fc.in_features
num_classes = 12
new_fc = nn.Linear(in_features, num_classes)
model.fc = new_fc
model = model.cuda()

# loss and optimizer
loss_function = nn.CrossEntropyLoss()
loss_f = loss_function.cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


# 开始训练
epoch = 20
for i in range(epoch):
    pe_train_acc = 0
    model.train()
    for batch_id, data in enumerate(train_dataloader):
        features, labels = data
        features = features.cuda()
        labels = labels.cuda()

        predicts = model(features)
        loss = loss_f(predicts, labels)
        acc = (predicts.argmax(1) == labels).sum()
        pe_train_acc = pe_train_acc + acc

        loss.backward()
        opt.step()
        opt.zero_grad()

        print('epoch: {}, batch: {}, loss: {}'.format(i + 1, batch_id + 1, loss.item()))

    print("-------epoch: {}, acc: {}-------".format(i + 1, pe_train_acc/train_data_size))

    model.eval()
    print("--------eval--------")
    acc = 0
    total_test_loss = 0
    total_right = 0
    with torch.no_grad():
        for data in valid_dataloader:
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()

            predicts = model(imgs)
            loss = loss_f(predicts, labels)
            right = (predicts.argmax(1) == labels).sum()
            total_test_loss = total_test_loss + loss
            total_right = total_right + right

        total_acc = total_right / valid_data_size
        print("total_loss: {}".format(total_test_loss))
        print("total_acc: {}".format(total_acc))

torch.save(model.state_dict(), "resnet_to_cat12.pth")




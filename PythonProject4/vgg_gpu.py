#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

# ---------- 1. 基本配置 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 5
SAVE_DIR = "ckpt"
LOG_DIR = "runs"
os.makedirs(SAVE_DIR, exist_ok=True)

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# ---------- 2. GPU 增强 ----------
from torchvision.transforms import v2

# 确保所有图像的尺寸一致，调整为224x224
gpu_transform = v2.Compose([
    v2.Resize((224, 224)),  # 确保图像的尺寸一致
    v2.RandomHorizontalFlip(),
    v2.Normalize(mean, std)  # 接收 float tensor，返回标准化 tensor
])

# ---------- 3. 数据集 ----------
# 只做 ToTensor，不做任何增强 / 标准化
cpu_tf = torchvision.transforms.ToTensor()

# 更新为 flower_data 数据集路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # 获取数据根路径
image_path = os.path.join(data_root, "data_set", "flower_data")  # flower_data 数据集路径
assert os.path.exists(image_path), f"{image_path} path does not exist."

train_set_raw = torchvision.datasets.ImageFolder(
    root=os.path.join(image_path, "train"), transform=cpu_tf)
test_set_raw = torchvision.datasets.ImageFolder(
    root=os.path.join(image_path, "val"), transform=cpu_tf)

# 把原始数据集包装成“GPU 增强”数据集
class GPUAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, gpu_tfm):
        self.dataset = dataset
        self.gpu_tfm = gpu_tfm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]   # img: [3,32,32] 0-1 tensor
        img = img.to(DEVICE)             # 先上 GPU
        img = self.gpu_tfm(img)          # GPU 内增强 + 标准化
        return img, label

train_set = GPUAugmentDataset(train_set_raw, gpu_transform)
test_set = GPUAugmentDataset(test_set_raw, v2.Normalize(mean, std))  # 测试只标准化

# DataLoader：数据已在 GPU，无需多进程
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=False)

# ---------- 4. 模型 ----------
def build_resnet18(num_classes=5):
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

# ---------- 5. 训练 / 测试 ----------
def train_one_epoch(net, criterion, optimizer, epoch, writer):
    net.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:   # 已 GPU+标准化
        optimizer.zero_grad()
        outs = net(imgs)
        labels = labels.to(DEVICE, non_blocking=True)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outs.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = correct / total
    avg_loss = total_loss / total
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/acc", acc, epoch)
    print(f"Epoch {epoch:3d} | train loss {avg_loss:.4f} acc {acc:.4f}")

@torch.no_grad()
def evaluate(net, criterion, epoch, writer, best_acc):
    net.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        outs = net(imgs)
        loss = criterion(outs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outs.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = correct / total
    avg_loss = total_loss / total
    writer.add_scalar("test/loss", avg_loss, epoch)
    writer.add_scalar("test/acc", acc, epoch)
    print(f"          | test  loss {avg_loss:.4f} acc {acc:.4f} (best {best_acc:.4f})")
    return max(acc, best_acc)

# ---------- 6. main ----------
def main():
    net = build_resnet18(num_classes=5).to(DEVICE)  # 5 classes for flower data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    writer = SummaryWriter(LOG_DIR)
    best_acc = 0.0

    print("开始训练（GPU 增强）...")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(net, criterion, optimizer, epoch, writer)
        best_acc = evaluate(net, criterion, epoch, writer, best_acc)
        scheduler.step()
        if epoch % 50 == 0 or epoch == EPOCHS:
            torch.save(net.state_dict(), os.path.join(SAVE_DIR, f"epoch{epoch}.pth"))
    writer.close()
    print(f"Finished! best_acc={best_acc:.4f}  cost={(time.time()-start)/60:.1f} min")

if __name__ == "__main__":
    main()

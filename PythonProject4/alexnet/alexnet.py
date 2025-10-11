import json
import os
import time
import argparse
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from tqdm import tqdm

# ---------------- 超参数 -----------------
def get_args():
    parser = argparse.ArgumentParser(description='Train AlexNet on Flower5')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data', type=str, default='data_set/flower_data')
    parser.add_argument('--save', type=str, default='AlexNet_1.pth')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

args = get_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

# ---------------- 数据 -------------------
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

image_path = os.path.join(args.data)
train_set = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                 transform=data_transform['train'])
val_set = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                               transform=data_transform['val'])
class_idx = train_set.class_to_idx
json_str = json.dumps({v: k for k, v in class_idx.items()}, indent=4)
with open('../class_indices.json', 'w') as f:
    f.write(json_str)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set,
                                         batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

# ---------------- 模型 -------------------
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

net = AlexNet(num_classes=5, init_weights=True).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ---------------- 训练/验证 -------------------
best_acc = 0.0
t1 = time.perf_counter()

for epoch in range(args.epochs):
    # ---------- train ----------
    net.train()
    print('Using device:', device)
    running_loss = 0.0
    train_bar = tqdm(train_loader, ncols=100)
    for images, labels in train_bar:  # 直接解包
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.set_description(f'Epoch[{epoch+1}/{args.epochs}] loss:{loss.item():.3f}')
    # scheduler.step()

    # ---------- validate ----------
    net.eval()
    correct = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, ncols=100)
        for images, labels in val_bar:  # 直接解包
            images, labels = images.to(device), labels.to(device)
            outputs = net(images.to(device))
            correct += (torch.max(outputs, dim=1)[1] == labels).sum().item()
    val_acc = correct / len(val_set)
    print(f'Epoch[{epoch+1}/{args.epochs}]  '
          f'train_loss: {running_loss/len(train_loader):.3f}  '
          f'val_accuracy: {val_acc:.3f}  '
          f'time: {time.perf_counter()-t1:.1f}s')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), args.save)
        print(f'*** best model updated: {val_acc:.3f} ***')

print('Finished Training')
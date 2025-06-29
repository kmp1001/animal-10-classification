import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import cos, pi
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import RandAugment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print("Using device:", device)

# Using RandAugment and ImageNet std
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # moderately retain random erasure, set these parameters carefully ( you can try for many times)
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# you should change the route here, don;t forget to download the animal_10 datasets first~~
train_dataset = datasets.ImageFolder(root='data/animal_10/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='data/animal_10/val', transform=val_test_transform)
test_dataset = datasets.ImageFolder(root='data/animal_10/test', transform=val_test_transform)

print("Dataset sizes:",
      f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

# ========== DataLoader ==========
batch_size = 64  # if the memory is insufficient, it can be changed~
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"train_loader={len(train_loader)}, val_loader={len(val_loader)}, test_loader={len(test_loader)}")


# MixUp/CutMix helper functions
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0)).to(x.device)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2))
    return x, y_a, y_b, lam


# ECA module
class ECAAttention(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


def conv_bn(inp, oup, kernel_size, stride=1, activation=nn.Hardswish):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, (kernel_size // 2), bias=False),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )

# InvertedResidualECA module
class InvertedResidualECA(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False):
        super(InvertedResidualECA, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1 and inp == oup)

        layers = []
        activation = nn.Hardswish
        if expand_ratio != 1:
            layers.append(conv_bn(inp, hidden_dim, 1, 1, activation))
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(activation(inplace=True))

        if use_eca:
            layers.append(ECAAttention(hidden_dim))

        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        else:
            return out


# improved network(I just use ECA in the later layers & adding Dropout)
class ImprovedAnimalNetNoFusion_NoArc(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedAnimalNetNoFusion_NoArc, self).__init__()
        self.stem = conv_bn(3, 16, 3, 2)

        block_setting = [
            # t,  c,   n,  s,   use_eca
            (1, 16, 1, 1, False),
            (4, 24, 2, 2, False),
            (4, 32, 3, 2, False),
            (4, 64, 3, 2, True),
            (6, 96, 2, 1, True),
            (6, 128, 2, 2, True),
        ]

        input_channel = 16
        features = []
        for t, c, n, s, eca in block_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                block = InvertedResidualECA(input_channel, c, stride, t, use_eca=eca)
                features.append(block)
                input_channel = c
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)  # 新增Dropout
        self.classifier = nn.Linear(input_channel, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# LabelSmoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, target):
        num_classes = preds.size(-1)
        log_probs = self.log_softmax(preds)
        with torch.no_grad():
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# Train & Validation
def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch,
                    mixup_alpha=0.2, cutmix_alpha=0.0, scaler=None):
    model.train()
    epoch_train_loss = 0.0
    loader_tqdm = tqdm(dataloader, desc=f"Training Epoch[{epoch}]", leave=False)

    for images, labels in loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with amp.autocast(enabled=(scaler is not None)):
            # choose randomly
            if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
                if random.random() < 0.5:
                    images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                else:
                    images, y_a, y_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
            elif mixup_alpha > 0.0:
                images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
            elif cutmix_alpha > 0.0:
                images, y_a, y_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
            else:
                lam = 1.0
                y_a, y_b = labels, labels

            outputs = model(images)
            loss = lam * loss_fn(outputs, y_a) + (1 - lam) * loss_fn(outputs, y_b)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_train_loss += loss.item() * images.size(0)
        loader_tqdm.set_postfix(loss=loss.item())

    epoch_train_loss /= len(dataloader.dataset)
    return epoch_train_loss


def val_one_epoch(model, dataloader, loss_fn, epoch):
    model.eval()
    epoch_val_loss = 0.0
    preds_list = []
    labels_list = []

    loader_tqdm = tqdm(dataloader, desc=f"Validating Epoch[{epoch}]", leave=False)
    with torch.no_grad():
        for images, labels in loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            with amp.autocast():
                outputs = model(images)
                val_loss = loss_fn(outputs, labels)

            epoch_val_loss += val_loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            preds_list.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            loader_tqdm.set_postfix(loss=val_loss.item())

    epoch_val_loss /= len(dataloader.dataset)
    val_acc = accuracy_score(labels_list, preds_list)
    return epoch_val_loss, val_acc


# model example
model = ImprovedAnimalNetNoFusion_NoArc(num_classes=10).to(device)

# optimizer & schedule(here I use AdamW & Cosine)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

# Early stopping
patience = 10
best_val_loss = np.inf
epochs_no_improve = 0

# Mix accuracy
scaler = amp.GradScaler()

epochs = 80
training_loss_history = []
validation_loss_history = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    train_loss = train_one_epoch(
        model, train_loader, optimizer, loss_fn,
        epoch=epoch,
        mixup_alpha=0.2,
        cutmix_alpha=0.0,
        scaler=scaler
    )
    training_loss_history.append(train_loss)

    val_loss, val_acc = val_one_epoch(model, val_loader, loss_fn, epoch=epoch)
    validation_loss_history.append(val_loss)
    scheduler.step()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
plt.figure(figsize=(10, 5))
plt.plot(training_loss_history, label='Training Loss')
plt.plot(validation_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

def inference_with_tta(model, images, n_augment=5):
    model.eval()
    final_logits = None
    with torch.no_grad():
        for _ in range(n_augment):
            aug_images = images.clone()
            flip_mask = (torch.rand(aug_images.size(0)) > 0.5).to(device)
            for i in range(aug_images.size(0)):
                if flip_mask[i]:
                    aug_images[i] = torch.flip(aug_images[i], dims=[2])
            outputs = model(aug_images)
            if final_logits is None:
                final_logits = outputs
            else:
                final_logits += outputs
    final_logits = final_logits / n_augment
    return final_logits


# evaluation on test datasets
best_model = ImprovedAnimalNetNoFusion_NoArc(num_classes=10).to(device)
best_model.load_state_dict(torch.load('best_model.pth'))
best_model.eval()

predicted_labels = []
actual_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = inference_with_tta(best_model, images, n_augment=5)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(actual_labels, predicted_labels, average='weighted', zero_division=0)

print(f"test accuracy:  {accuracy * 100:.2f}%")
print(f"test precision:  {precision * 100:.2f}%")
print(f"test recall:  {recall * 100:.2f}%")
print(f"test F1-score: {f1 * 100:.2f}%\n")

print(classification_report(actual_labels, predicted_labels, zero_division=0))

cm = confusion_matrix(actual_labels, predicted_labels)
class_names = test_dataset.classes
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title('Test Set with TTA')
plt.xlabel('predicted labels')
plt.ylabel('true labels')
plt.show()

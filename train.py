import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms

from dataset import VOCDataset
from model import PreTrained, YOLOv1
from loss import YoloLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
EPOCHS = 10

# Data Loader - Cifar Dataset
# https://www.kaggle.com/code/mainscientist/cifar-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

aug_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset_aug_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=aug_transform)
train_dataset_cifar = ConcatDataset([dataset_cifar, dataset_aug_cifar])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Data Loader - Pascal VOC Dataset
# transform = 
train_dataset_voc = VOCDataset(csv_file = "C:/data/cifar", transform=None)
test_dataset_voc = VOCDataset(csv_file = "C:/data/cifar", transform=None)
# train_dataloader_voc = DataLoader(train_dataset_voc, batch_size=16, shuffle=True, num_workers=4)
# test_dataloader_voc = DataLoader(test_dataset_voc, batch_size=16, shuffle=True, num_workers=4)

# Reset model
pretrained_model = PreTrained().to(device)
model = YOLOv1().to(device)

# Define loss function and optimizer
criterion = YoloLoss()
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# training loop
for epoch in range(EPOCHS):
    pass
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
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

aug_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_voc = transforms.Compose([
    [transforms.Resize((448, 448)), transforms.ToTensor(),]
])

dataset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset_aug_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=aug_transform)
train_dataset_cifar = ConcatDataset([dataset_cifar, dataset_aug_cifar])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Data Loader - Pascal VOC Dataset
# transform = 
train_dataset_voc = VOCDataset(csv_file = "C:/data/pascalvoc", transform=transform_voc)
test_dataset_voc = VOCDataset(csv_file = "C:/data/pascalvoc", transform=transform_voc)
# cpu를 사용하므로 batch_size를 1로 설정
train_dataloader_voc = DataLoader(train_dataset_voc, batch_size=1, shuffle=True, num_workers=4)
test_dataloader_voc = DataLoader(test_dataset_voc, batch_size=1, shuffle=True, num_workers=4)

# Reset model
pretrained_model = PreTrained().to(device)
model = YOLOv1(pretrained_model, split_size=7, num_boxes=2, num_classes=20).to(device)

# Define loss function and optimizer
criterion = YoloLoss()
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
#optim_momentum = optim.SGD(model.parameters(),lr=LEARNING_RATE, momentum=0.9)



# training loop
for epoch in range(EPOCHS):
    pass
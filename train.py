import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms

from dataset import VOCDataset
from model import YOLOv1 #PreTrained,
from loss import YoloLoss
from tqdm import tqdm
import numpy as np
import torchvision.models as models

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
    transforms.Resize((448, 448)), transforms.ToTensor()
])



def train(train_loader, model, optimizer, criterion, ispretrained=False):
    train_loss = []

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in tqdm(train_loader):
            #int형 변수에 tensor 함수를 사용할 수 없어서 에러 발생
            #int형 변수를 tensor형으로 변환하여 적용해줌.
            tensor_labels = torch.tensor(labels)
            x, y = images.to(device), labels.to(device)
            #print('확인용: ', y)



            output = model(x)
            print('output 크기:',output.shape)
            print('output:', output)
            print('y 크기:',y.shape)
            print('y:', y)
            if(not ispretrained):
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            else:
                criterion = nn.CrossEntropyLoss()
                output_ = torch.tensor(output.to(torch.float32), requires_grad=True)
                y_ = torch.tensor(y.to(torch.float32), requires_grad=True)
                loss = criterion(output_,y_)
                optimizer.zero_grad()
                print('loss:', loss)
                loss.backward()
                optimizer.step()
                train_loss.append(loss)

        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}]')

    return model, optimizer, criterion


def main():
    ## PreTrain
    # Reset model
    #pretrained_model = PreTrained().to(device)

    # Define loss function and optimizer
    criterion_p = YoloLoss()
    # optimizer_p = optim.Adam(
    #     pretrained_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    # )
    # optim_momentum = optim.SGD(model.parameters(),lr=LEARNING_RATE, momentum=0.9)

    # Define dataset
    # path_cifar = 'C:/data/cifar/'
    # dataset_cifar = torchvision.datasets.CIFAR10(root=path_cifar, train=True, download=False, transform=transform)
    # dataset_aug_cifar = torchvision.datasets.CIFAR10(root=path_cifar, train=True, download=False, transform=aug_transform)
    # train_dataset_cifar = ConcatDataset([dataset_cifar, dataset_aug_cifar])
    # test_dataset_cifar = torchvision.datasets.CIFAR10(root=path_cifar, train=False, download=False, transform=transform)
    #
    # #print('cifar: ', train_dataset_cifar.__getitem__(0))
    #
    # # Define dataloader - cifar
    # train_dataloader_cifar = DataLoader(train_dataset_cifar, batch_size=32, shuffle=True, num_workers=2)
    # test_dataloader_cifar = DataLoader(test_dataset_cifar, batch_size=32, shuffle=True, num_workers=2)
    #
    # # Training train(train_loader, model, optimizer, criterion):
    # pretrained_model, criterion_p, optimizer_p = train(train_dataloader_cifar, pretrained_model, optimizer_p, nn.CrossEntropyLoss(), ispretrained=True)


    ## YOLOv1 Train
    # Reset model
    pretrained_model = models.vgg16()
    model = YOLOv1(pretrained_model, split_size=7, num_boxes=2, num_classes=20).to(device)

    # Define loss function and optimizer
    criterion = YoloLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # optim_momentum = optim.SGD(model.parameters(),lr=LEARNING_RATE, momentum=0.9)

    # Define dataset
    train_dataset_voc = VOCDataset(csv_file = "C:/data/pascalvoc/train.csv", transform=transform_voc)
    test_dataset_voc = VOCDataset(csv_file = "C:/data/pascalvoc/test.csv", transform=transform_voc)
    #x, y = train_dataset_voc.__getitem__(0)
    #x, y = torch.tensor(x), torch.tensor(y)
    #print(x.shape)
    #print(y.shape)


    # Define dataloader - Pascal VOC Dataset (we use cpu, so batch_size is 1)
    train_dataloader_voc = DataLoader(train_dataset_voc, batch_size=1, shuffle=True, num_workers=4)
    test_dataloader_voc = DataLoader(test_dataset_voc, batch_size=1, shuffle=True, num_workers=4)

    # Training
    model, criterion, optimizer = train(train_dataloader_voc, model, optimizer, criterion)

if __name__ == "__main__":
    main()
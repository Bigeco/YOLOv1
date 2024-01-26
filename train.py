import torch
import torch.nn as nn
import torch.optim as optim

from model import PreTrained, YOLOv1
from loss import YoloLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
EPOCHS = 10

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
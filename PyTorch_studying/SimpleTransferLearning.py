#  Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
#  Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  Load model

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


model = torchvision.models.vgg16(pretrained=True)
model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)
model.to(device)

#  accuracy check function
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()


#  Hyper parameters
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5

#  Load data

train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#  Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#  Train network

for epoch in range(num_epochs):
    for batch_index, (data, targets) in enumerate(train_loader):
        #  Data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        #  Forward
        scores = model(data)
        loss = criterion(scores, targets)

        #  Backward
        optimizer.zero_grad()
        loss.backward()

        #  Gradient descent
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}')
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)

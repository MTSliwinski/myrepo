#  Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#  Create fully connected network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


#  accuracy check function
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


#  Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  Hyper parameters

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

#  Load data

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#  Initialize network

model = NN(input_size=input_size, num_classes=num_classes).to(device)

#  Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#  Train network

for epoch in range(num_epochs):
    for batch_index, (data, targets) in enumerate(train_loader):
        #  Data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        #  Get to correct shape
        data = data.reshape(data.shape[0], -1)

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

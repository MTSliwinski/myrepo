#Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice and easy way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
#Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size,1000)
        self.fc2 = nn.Linear(500, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(1600, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


#Check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameteres
learning_rate = 0.001
batch_size = 64
num_epochs = 5

#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Initialize network
model = CNN().to(device=device)


#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


#Train Network

for epoch in range(num_epochs):
    for batch_ifx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores,targets)
        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradent descent or adam step
        optimizer.step()
    print(f'Accuracy for epoch {epoch + 1} / {num_epochs}')
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)






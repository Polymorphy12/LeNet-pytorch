# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1),padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84,10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_channel = 1
num_classes = 10
learning_rate = 0.003
batch_size = 64
num_epochs = 4

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, download=True, transform=transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]
))
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, download=True, transform=transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]
))
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle=True)

# Initialize Network
model = LeNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for idx, (data, targets) in enumerate(train_loader):
        #Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        #forward
        scores = model(data)
        loss = criterion(scores,targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent
        optimizer.step()


#Check accuracy on training & test set to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    # you don't have to calculate gradients while you're evaluating
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
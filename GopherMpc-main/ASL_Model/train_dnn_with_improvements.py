

# Import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = "cpu"

# Define model
# Code reused from PyTorch tutorial:
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(39, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 24),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

#Residual block for better gradient flow
class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.PReLU(),
            nn.Linear(output_size, output_size),
            nn.BatchNorm1d(output_size),
        )
        self.shortcut = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

#Add Residual Block to NN
#Allows better gradient flow during backpropagation
#Prevents vanishing gradient problem 
#Parametric ReLU (PReLU)
#  Learnable activation function that can adapt better to the data
#Gradual Feature Compression
# Reduced neoron counts in stepwise fashion (256-128-64-24) to make it
# efficient and focused
#Batch Normalization
#  Applied after every dense layer to stablizie training and reduce th
#  need for careful initilization
class ResImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(39, 256),   # Input layer
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.3),
            
            ResidualBlock(256, 256),  # Residual block
            nn.Dropout(0.3),

            nn.Linear(256, 128),  # Transition layer
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),   # Compression layer
            nn.BatchNorm1d(64),
            nn.PReLU(),

            nn.Linear(64, 24),    # Output layer
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


#Column A Improvement 
#Deeper Network, Add batch normalization, dropout
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(39, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 24),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



# Code edited from PyTorch tutorial site:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class MediaPipePoseDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.positions = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):

        return len(self.positions)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.positions.iloc[idx, 1:]
        data = np.array(data)
        data = data.astype('float').reshape(len(data), -1)
        data = np.transpose(data)
        data = torch.from_numpy(data)
        data = data.type(torch.float32)

        character = self.positions.iloc[idx, 0]
        character_int = 0

        if character=='A':
            character_int = 0
        elif character=='B':
            character_int = 1
        elif character=='C':
            character_int = 2
        elif character=='D':
            character_int = 3
        elif character=='E':
            character_int = 4
        elif character=='F':
            character_int = 5
        elif character=='G':
            character_int = 6
        elif character=='H':
            character_int = 7
        elif character=='I':
            character_int = 8
        elif character=='K':
            character_int = 9
        elif character=='L':
            character_int = 10
        elif character=='M':
            character_int = 11
        elif character=='N':
            character_int = 12
        elif character=='O':
            character_int = 13
        elif character=='P':
            character_int = 14
        elif character=='Q':
            character_int = 15
        elif character=='R':
            character_int = 16
        elif character=='S':
            character_int = 17
        elif character=='T':
            character_int = 18
        elif character=='U':
            character_int = 19
        elif character=='V':
            character_int = 20
        elif character=='W':
            character_int = 21
        elif character=='X':
            character_int = 22
        elif character=='Y':
            character_int = 23

        sample = data, character_int

        if self.transform:
            sample = self.transform(sample)

        return sample
    
def train(dataloader, model, loss_fn, optimizer):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct

def main():

    #Load network on to either GPU or CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using {device} device")

    #Create DNN network
    #model = ImprovedNeuralNetwork()
    model = ResImprovedNeuralNetwork()
    #model = NeuralNetwork()

    #Loss Function and Optimizer
    loss_func = nn.CrossEntropyLoss()

    #Optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    #Learning Rate Scheduler: Reduce LR every 5 epochs by a factor of 0.1
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    #Move DNN to GPU or CPU
    model = model.to(device)
    print(model)

    #Load data
    batch_size = 8
    training_dataset = MediaPipePoseDataset('Synthetic_Train_Features2.csv')
    training_dataloader = DataLoader(training_dataset, batch_size, shuffle=True)

    testing_dataset = MediaPipePoseDataset('Synthetic_Test_Features2.csv')
    testing_dataloader = DataLoader(testing_dataset, batch_size, shuffle=True)

    #Train and validate network
    epochs = 20
    test_loss = 0.0
    correct = 0.0
    running_loss = np.array([0.0])
    running_correct = np.array([0.0])
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_dataloader, model, loss_func, optimizer)
        test_loss, correct = test(testing_dataloader, model, loss_func)

        running_loss = np.append(running_loss, test_loss)
        running_correct = np.append(running_correct, correct)

        #Step the Scheduler
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
    print("Done!")

    #Save network
    torch.save(model.state_dict(), "Synthetic_Features2.pth")

    #Plot Results
    epoch_range = range(1,epochs+1,1)
    running_loss = np.delete(running_loss, 0, axis=0)
    running_correct = np.delete(running_correct, 0, axis=0)

    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.plot(epoch_range, running_loss, color='tab:blue')
    # ax1.set_title('Average Loss per Epoch')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('% Loss')

    # ax2.plot(epoch_range, running_correct*100, color='tab:orange')
    # ax2.set_title('Accuracy per Epoch')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('% Accuracy')

    # plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plotting loss
    ax1.plot(epoch_range, running_loss, color='tab:blue')
    ax1.set_title('Average Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('% Loss')

    # Plotting accuracy
    ax2.plot(epoch_range, running_correct * 100, color='tab:orange')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('% Accuracy')

    # Save the figure to a file
    plt.savefig("training_results.png", dpi=300, bbox_inches='tight')

    # Optional: close the plot to free memory
    plt.close(fig)



if __name__ == "__main__":
    main()
# mnist_mlp.py
# Train a simple multilayer perceptron (MLP) to classify handwritten digits (MNIST).
# Based on the official PyTorch quickstart tutorial, adapted from FashionMNIST to MNIST.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ---------------------------------------------------------------
# Hyperparameters: change these to run experiments
# ---------------------------------------------------------------
learning_rate = 1e-1
batch_size = 64
epochs = 2

# ---------------------------------------------------------------
# Data: download MNIST and wrap it in DataLoaders.
# Each image is 28x28 grayscale. ToTensor converts it to a tensor
# and scales pixel values from 0-255 down to 0-1.
# ---------------------------------------------------------------
training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

# DataLoaders split the data into batches. Training data is shuffled each epoch.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# ---------------------------------------------------------------
# Device: use the GPU if one is available, otherwise the CPU
# ---------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# ---------------------------------------------------------------
# Model: flatten each 28x28 image into 784 values, then pass it
# through fully connected layers with ReLU activations.
# The final layer outputs 10 scores, one per digit (0-9).
# ---------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# ---------------------------------------------------------------
# Loss and optimizer: CrossEntropyLoss compares the 10 scores to the
# true label. SGD adjusts the weights to reduce that loss.
# ---------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------
# Training: for each batch, predict, compute the loss, compute
# gradients, and update the weights. Returns the average loss.
# ---------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass: compute predictions and the loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass: clear old gradients, compute new ones, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------------------------------------------------------------
# Testing: run the model on unseen data without updating weights.
# argmax picks the digit with the highest score as the prediction.
# ---------------------------------------------------------------
def test(dataloader, model, loss_fn):
    model.eval()
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    test_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return test_loss, accuracy

# ---------------------------------------------------------------
# Main loop: train for the set number of epochs, testing after each
# ---------------------------------------------------------------
for epoch in range(epochs):
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    test_loss, accuracy = test(test_dataloader, model, loss_fn)
    print(f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss:.4f} | "
          f"test loss: {test_loss:.4f} | test accuracy: {accuracy * 100:.2f}%")

# ---------------------------------------------------------------
# Save the trained weights so the model can be reloaded later
# ---------------------------------------------------------------
torch.save(model.state_dict(), "mnist_mlp.pth")
print("Saved model to mnist_mlp.pth")
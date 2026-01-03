import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. DATA PREPARATION
# -------------------
# Define a "transform" to convert images to PyTorch Tensors and normalize them
# Normalization (0.5, 0.5) shifts pixel values from [0, 1] to [-1, 1] for better training stability
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader
# This handles shuffling and batching (loading 64 images at a time)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# 2. MODEL SETUP
# --------------
# (Re-using the SimpleCNN class from the previous step)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Model, Loss Function, and Optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()  # Standard loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam is a popular, efficient optimizer


# 3. TRAINING LOOP
# ----------------
print("Starting Training...")
num_epochs = 3 # How many times we go through the entire dataset

for epoch in range(num_epochs):
    running_loss = 0.0
    
    # Iterate over the DataLoader (batch by batch)
    for i, (images, labels) in enumerate(train_loader):
        
        # A. Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels) # Calculate error
        
        # B. Backward Pass and Optimization
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Calculate gradients (backpropagation)
        optimizer.step()      # Update weights
        
        running_loss += loss.item()
        
        # Print progress every 300 batches
        if (i+1) % 300 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/300:.4f}')
            running_loss = 0.0

print("Training Finished!")
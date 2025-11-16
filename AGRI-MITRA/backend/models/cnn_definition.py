import torch
import torch.nn as nn

# This is the same class definition from your 'ml/train_disease_cnn.py'
# By placing it here, your backend can access it locally without
# trying to import from the 'ml' folder.

class SimpleCNN(nn.Module):
    """A simple CNN for demonstration."""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # 3 input channels (RGB), 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 224x224 -> 112x112
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 112x112 -> 56x56
        
        # Calculate the flattened size: 32 channels * 56x56 pixels
        self.fc1 = nn.Linear(32 * 56 * 56, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # (Batch, 3, 224, 224)
        x = self.pool(self.relu(self.conv1(x)))
        # (Batch, 16, 112, 112)
        x = self.pool(self.relu(self.conv2(x)))
        # (Batch, 32, 56, 56)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 32 * 56 * 56) 
        # (Batch, 100352)
        
        x = self.relu(self.fc1(x))
        # (Batch, 128)
        x = self.fc2(x)
        # (Batch, num_classes)
        return x
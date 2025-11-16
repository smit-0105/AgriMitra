import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# --- Configuration ---
NUM_CLASSES = 10  # TODO: Change this to your number of disease classes
DATA_DIR = 'data/' # TODO: Put your 'train' and 'val' image folders here
MODEL_SAVE_PATH = 'disease_model.pt'
NUM_EPOCHS = 15

class SimpleCNN(nn.Module):
    """A simple CNN for demonstration."""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128) # Assuming 224x224 input -> 112 -> 56
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    print("--- Starting Disease Model Training ---")
    
    # TODO: Create real DataLoaders
    # This is a mock dataloader. You should use datasets.ImageFolder
    # with your 'data/train' and 'data/val' directories.
    # Example transform:
    # data_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    print(f"Loading data from {DATA_DIR} (mock)...")
    # Mock data
    train_loader = [(torch.rand(8, 3, 224, 224), torch.randint(0, NUM_CLASSES, (8,))) for _ in range(10)]
    
    model = SimpleCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    print("--- Training Finished ---")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"IMPORTANT: Copy {MODEL_SAVE_PATH} to backend/models/disease_model.pt")

if __name__ == "__main__":
    train_model()
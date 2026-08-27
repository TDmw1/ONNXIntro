import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

# 1. Define the model architecture
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 output classes for digits 0-9
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# 2. Download and load the MNIST dataset
print("Downloading dataset...")
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Set up the model, loss function, and optimizer
model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
print("Starting training (this takes about 10-20 seconds)...")
for images, labels in train_loader:
    optimizer.zero_grad()         # Clear old gradients
    outputs = model(images)       # Forward pass
    loss = criterion(outputs, labels) # Calculate error
    loss.backward()               # Backward pass
    optimizer.step()              # Update weights

# 5. Save the standard PyTorch format
torch.save(model.state_dict(), "mnist_model.pt")
print("Training complete! PyTorch weights saved to mnist_model.pt")
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Set True to train the model.
# Set False to load the saved model.

TRAIN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.ToTensor()

train_dataset = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# CNN Model

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            # After second MaxPool:
            # (batch_size, 16, 7, 7)
            # Flatten -> 16 × 7 × 7 = 784
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


model = SimpleCNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAINING

if TRAIN:

    epochs = 10
    loss_history = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    plt.figure(figsize=(6,4))
    plt.plot(loss_history, marker="o", color="red", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluation
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "simple_cnn_mnist.pth")

    print("Model saved as simple_cnn_mnist.pth")

# LOAD SAVED MODEL

else:

    model.load_state_dict(
        torch.load("simple_cnn_mnist.pth", map_location=device)
    )

    model.eval()

    print("Model loaded successfully!")

# RANDOM PREDICTION

index = random.randint(0, len(test_dataset) - 1)

image, true_label = test_dataset[index]

image_batch = image.unsqueeze(0).to(device)

with torch.no_grad():

    output = model(image_batch)

    predicted_label = output.argmax(dim=1).item()

    confidence = torch.softmax(output, dim=1).max().item() * 100


plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual: {true_label} | Predicted: {predicted_label}")
plt.axis("off")
plt.show()

print(f"Image Index : {index}")
print(f"Actual      : {true_label}")
print(f"Predicted   : {predicted_label}")
print(f"Confidence  : {confidence:.2f}%")
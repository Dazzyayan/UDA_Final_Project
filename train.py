import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time


def load_data(data_dir, batch_size=32, train_split=0.8):
    """
    Loads preprocessed images from disk to memory
    We are expecting preprocessed images which have already been normalised
    Images should be in folders labelling their class
    """

    # Resize images to 224x224 to save memory (this is a hotfix to fix memory issues quickly)
    # Resizing the images does not make sense given that we have already preprocessed the images
    # However the system begins to run out of ram, so this is a compromise solution taken under time constraints
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Test Train Split
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Creating dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Loaded {len(dataset)} images. Train: {train_size}, Test: {test_size}")
    return train_loader, test_loader, dataset.classes


class PhotographClassifier(torch.nn.Module):
    def __init__(self):
        super(PhotographClassifier, self).__init__()

        # First block: CONV => RELU => BN => CONV => RELU => BN => POOL
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),  # We perform this in place to save a little memory
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second block: CONV => RELU => BN => CONV => RELU => BN => POOL
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third block: CONV => RELU => BN => CONV => RELU => BN => POOL
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully connected layers for classification
        # Classifier: [FC => RELU => BN] * 2 => FC
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 2)  # narrowed down to 2 final classes
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def train(model, train_loader, criterion, optimizer, device):
    # This method trains the model for one epoch
    model.train()
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)  # Loading data to GPU
        optimizer.zero_grad()  # Zero gradients each run
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)  # Extract prediction
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Train Accuracy: {acc:.2f}%")
    return acc


def validate(model, test_loader, device):
    # Validate the model performance
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc


def execute():
    device = torch.device('cuda')  # please switch to cpu if you do not have gpu

    # Load data
    data_dir = r'F:\MLDS\UDA Datasets\preprocessed_1800'
    train_loader, test_loader, classes = load_data(data_dir)

    # Initialize model, loss, and optimizer
    model = PhotographClassifier(num_classes=len(classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001)  # We use ADAM optimiser as it is computational cost effective

    # Training loop
    num_epochs = 100  # We normally terminate (ctrl+c) the training before 100 epochs if we see no gains in accuracy
    best_val_acc = 0.0
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_acc = train(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, test_loader, device)

        # Save best model to disk
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with accuracy: {val_acc:.2f}%")

    print("Training complete. Exitting")


if __name__ == "__main__":
    execute()

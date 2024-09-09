import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.nn import Dropout
from imageNormalization import ChessDataset # Personal library
from datetime import datetime
from sklearn.model_selection import KFold

class ChessPieceModel(nn.Module):
    def __init__(self):
        super(ChessPieceModel, self).__init__()

        # Model layers:
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 13)  # 13 classes (6 pieces each team + empty squares)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
    
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
    
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

if __name__ == '__main__':

    # Record the start time
    start_time = datetime.now()

    IMAGES_PATH_TRAIN = ".\\Images_Split\\test"
    IMAGES_PATH_TEST = ".\\Images_Split\\train"

    # Transformation for the images (normalization)
    TRANSFORM = T.Compose([
        T.Resize((28, 28)),
        T.ToTensor()
    ])

    # Load the dataset
    dataset_train = ChessDataset(IMAGES_PATH_TRAIN, transform=TRANSFORM)
    dataset_test = ChessDataset(IMAGES_PATH_TEST, transform=TRANSFORM)[0:2000]
    #print(len(dataset_train))
    

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.0001

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model = ChessPieceModel().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}")

    # Record the end time
    end_time = datetime.now()
    print(f"Training took: {end_time - start_time}")

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

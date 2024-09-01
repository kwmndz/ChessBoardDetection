import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Dropout
from dataNormalization import ChessDataset # Personal library
from datetime import datetime

# Record the start time
start_time = datetime.now()

class ChessBoardModel(nn.Module):
    def __init__(self):
        super(ChessBoardModel, self).__init__() # Call the parent class intializer (nn.Module)

        # Model layers:
        # Conv2d: convolutional layer for 2d image processing
        # 3 input channels (RGB), 16 output channels, 3x3 kernel size (size of filter)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Fully connected layers (linear/dense layers)
        # Reducing the CNN 3 dimension tensors to a 1 dimension tensor
        # 56 * 56 is the size of the image after the convolutional layers
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        # outputs 8x8x13 tensor (8x8 board, 13 classes)
        self.fc2 = nn.Linear(512, 64 * 13) # 13 classes (6 pieces each team + empty squares)

    def forward(self, x):
        # Forward pass through the model
        # ReLU is a non-linear that introduces non linearity allowing the model to learn complex patterns
        x = F.relu(self.conv1(x))
        # Down sizes the size of the feature maps while keeping the important information
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        # second convolutional layer
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = x.view(x.size(0), -1) # flattens the tensor, -1 has the function infer it size by the amount of data
        
        # Fully connected layers (linear layers)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = x.view(-1, 8, 8, 13) # reshapes the tensor to the 8x8x13 format

        return x


if __name__ == '__main__':


    model = ChessBoardModel()
    # calculates the loss based on a function that quantifies the difference between the predicted and actual values
    # Uses these values to update the model's weights
    lossFunc = nn.CrossEntropyLoss() # Commonly used for multi-class classification 

    # Used to udpdate the model's weights based on the loss during training
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001, 
        weight_decay=0.001 # AKA L2 regularization, gives penalty for placing too much importance on one feature
    )


    # train the model off the cpu because I don't have a dedicated GPU
    model.to(torch.device("cpu"))
    #model.to(torch.device("cuda")) # Use if GPU is available and decent

    # Transformation for the images (normalization)
    TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(), # standard type for PyTorch
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    IMAGES_PATH_TRAIN = "./Images/NewTrain"
    IMAGES_PATH_TEST = "./Images/Test"

    # Load my child class of the Pytorch Dataset class
    dataset_train = ChessDataset(IMAGES_PATH_TRAIN, transform=TRANSFORM)
    dataset_test = ChessDataset(IMAGES_PATH_TEST, transform=TRANSFORM)

    print(dataset_train[0])
    data_loader_train = torch.utils.data.DataLoader( # iterates over the data 
        dataset_train, # dataset
        batch_size=32, # amount of samples to process in one iteration
        shuffle=True # shuffle 
    )
    print(data_loader_train)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=32,
        shuffle=True
    )

    # Prevent overfitting
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    # Dropout layer
    dropout = Dropout(p=0.5)

    # number of complete passes through the training dataset
    epochs = 10
    for epoch in range(epochs):

        model.train() # set model to training mode

        i=0
        for images, labels in data_loader_train:

            optimizer.zero_grad() # zero the gradients before running the model
            outputs = model(images) # forward pass
            labels = labels.view(-1, 8, 8) # reshapes the labels to the 8x8 format

            # .permute(0,3,1,2) changes the tensor from 4x8x8x13 to 4x13x8x8 to work with the loss func
            losses = lossFunc(outputs.permute(0,3,1,2), labels) # calculate the loss
            losses.backward() # backward pass
            optimizer.step() # update the weights

            """
            print(f"Epoch [{epoch+1}/{epochs}],", 
                  f"Step [{i+1}/{len(data_loader_train)}], Loss: {losses.item():.4f}")
            """
            #i+=1
        print(f"Epoch [{epoch+1}/{epochs}]",
                f"Time taken: {(datetime.now() - start_time).total_seconds()/60/60} seconds",
            )
        scheduler.step(losses) # update the learning rate based on the loss

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in data_loader_test:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=3)  # Get the predicted class for each square

            # Compare with labels to assess accuracy or other metrics
            correct = 0
            total = 0
            for pred, label in zip(predicted, labels):
                correct += (pred == label).sum().item()
                total += label.numel()

        accuracy = correct / total
        print(f"Accuracy: {accuracy:.2%}")

        if accuracy > 0.85:
            torch.save(model.state_dict(), "model.pth")
            print ("Model saved")

# Record the end time
end_time = datetime.now()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Program execution time: {elapsed_time.total_seconds()} seconds")

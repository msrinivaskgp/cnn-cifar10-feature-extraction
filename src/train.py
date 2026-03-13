import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os

from src.model import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# Hyperparameters
############################################

batch_size = 128
num_epochs = 40
learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-4
patience = 5

############################################
# Data Augmentation
############################################

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

############################################
# Evaluation Function
############################################

def evaluate(model, loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


############################################
# Training Function
############################################

def train_model():

    ############################################
    # Dataset
    ############################################

    full_train_dataset = CIFAR10(root='./data',
                                 train=True,
                                 download=True,
                                 transform=transform_train)

    test_dataset = CIFAR10(root='./data',
                           train=False,
                           download=True,
                           transform=transform_test)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(full_train_dataset,
                                              [train_size, val_size])

    ############################################
    # DataLoaders
    ############################################

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    ############################################
    # Model
    ############################################

    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=20,
                                    gamma=0.1)

    ############################################
    # Training
    ############################################

    best_val_accuracy = 0
    early_stop_counter = 0

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images,labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs,labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _,predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_accuracy = 100 * correct / total

        val_accuracy = evaluate(model,val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Loss:{running_loss/len(train_loader):.4f} "
              f"Train Acc:{train_accuracy:.2f}% "
              f"Val Acc:{val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            early_stop_counter = 0

        else:

            early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break

    ############################################
    # Load Best Model
    ############################################

    model.load_state_dict(best_model)

    ############################################
    # Test Accuracy
    ############################################

    test_accuracy = evaluate(model,test_loader)

    print("Final Test Accuracy:",test_accuracy,"%")

    ############################################
    # Save Model
    ############################################

    os.makedirs("outputs/models", exist_ok=True)

    torch.save(model.state_dict(),
               "outputs/models/best_model.pth")

    return model, train_loader
import os
from src.data.TrainClass import TrainDataset
from torch.utils.data import DataLoader
import torch


GLOBAL_PATH = os.getcwd()
IMGS_PATH = os.path.join(GLOBAL_PATH, 'src','data', 'release','train_images')
TRAIN_CSV_PATH = os.path.join(GLOBAL_PATH, 'src','data', 'release','train.csv')

train_dataset = TrainDataset(IMGS_PATH, TRAIN_CSV_PATH)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, data_loader, loss, optimizer, num_epochs, device):

    model = model.to(device)
    model.train() # metto il modello in modalità di addestramento

    loss_history     = []  # Initialize list to store loss per epoch
    accuracy_history = []  # Initialize list to store accuracy per epoch

    for epoch in range(num_epochs):

        total_loss = 0.0
        correct_pred = 0
        total_pred = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            optimizer.zero_grad()
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimizer.step()
            total_loss += loss.item()*images.size(0) # farò poi una media pesata perchè batch piccoli devono influenzare di meno la media

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)  # Get predicted class
            correct_pred += (predicted == labels).sum().item()  # Count correct predictions
            total_pred += labels.size(0) # Count total predictions

        # Compute metrics
        total_epoch_loss = total_loss / total_pred
        epoch_accuracy = correct_pred / total_pred
        loss_history.append(total_epoch_loss) # Append epoch loss
        accuracy_history.append(epoch_accuracy) # Append epoch accuracy
        print(f"Epoch {epoch + 1}, Loss: {total_epoch_loss}, Accuracy: {epoch_accuracy:.4f}")

    return loss_history, accuracy_history # Return history lists




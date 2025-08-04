import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm

# Import custom modules
from dataset import BCIDataset
from model import EEGNet

def parse_args():
    parser = argparse.ArgumentParser(description = "Train EEGNet model for motor imagery classification")
    parser.add_argument("--data_dir", type = str, default = "./../data/processed", help = "Directory containing the EEG data files")
    parser.add_argument("--epochs", type = int, default = 100, help = "Number of training epochs")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size for training")
    parser.add_argument("--learning_rate", type = float, default = 0.001, help = "Learning rate for the optimizer")
    parser.add_argument("--save-dir", type = str, default = "./../models", help = "Directory to save the best model")
    parser.add_argument("--save-file-name", type = str, default = "best_model.pth", help = "Filename for the saved model")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and Split Dataset
    dataset = BCIDataset(data_path = args.data_dir, dataset_type = "T")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Initialize Model, Loss Function, Optimizer
    model = EEGNet().to(device) # 22 channels, 4 classes, EEGNet 8,2 architecture
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training and Validation Loop
    best_val_accuracy = 0.0
    os.makedirs(args.save_dir, exist_ok = True) # Ensure save directory exists

    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc = f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({"loss": running_loss / total_train, "accuracy": 100 * correct_train / total_train})

        # Calculate post-epoch metrics
        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / total_train



        # Validation Phase
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc = f"Epoch {epoch + 1}/{args.epochs} [Val]")
            for inputs, labels in val_pbar:
                # Forward pass
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate metrics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Update progress bar
                val_pbar.set_postfix({"loss": running_loss / total_val, "accuracy": 100 * correct_val / total_val})

            # Calculate post-epoch metrics
            val_accuracy = 100 * correct_val / total_val
            val_loss = running_loss / total_val

            print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Save best model based on valiadation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_accuracy
                }, model_save_path)
                print(f"Best model saved to {model_save_path} with accuracy: {best_val_accuracy:.2f}%")

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import argparse
import os
import numpy as np
from tqdm import tqdm

# Import custom modules
from dataset import BCIDataset
from model import EEGNet

def parse_args():
    parser = argparse.ArgumentParser(description = "Train EEGNet model for motor imagery classification")
    parser.add_argument("--data-dir", type = str, default = "./../data/processed", help = "Directory containing the EEG data files")
    parser.add_argument("--epochs", type = int, default = 200, help = "Number of training epochs")
    parser.add_argument("--batch-size", type = int, default = 32, help = "Batch size for training")
    parser.add_argument("--learning-rate", type = float, default = 0.001, help = "Learning rate for the optimizer")
    parser.add_argument("--dropout-rate", type = float, default = 0.5, help = "Dropout rate for regularization")
    parser.add_argument("--weight-decay", type = float, default = 0.0, help = "Weight decay (L2 regularization) for the optimizer")
    parser.add_argument("--output-file", type = str, default = "./../models/cv_summary.csv")

    return parser.parse_args()

def run_fold(train_loader, val_loader, device, args):
    # Initialize model
    model = EEGNet(n_channels = 22, n_classes = 4, dropout_rate = args.dropout_rate).to(device) # EEGNet 8,2 architecture
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Early stopping parameters
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    patience = 15
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        
        # Validation Phase
        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_loss = running_val_loss / total_val

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    return best_val_accuracy

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CV for LR = {args.learning_rate}, WD = {args.weight_decay} on device: {device}")

    dataset = BCIDataset(data_path = args.data_dir, dataset_type = "T")

    kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(dataset), total = 10, desc = "Folds")):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size = args.batch_size, shuffle = True)
        val_loader = DataLoader(val_subset, batch_size = args.batch_size, shuffle = False)

        best_fold_accuracy = run_fold(train_loader, val_loader, device, args)
        fold_accuracies.append(best_fold_accuracy)
        print(f"Fold {fold + 1} | Best Validation Accuracy: {best_fold_accuracy:.2f}%")

    # Cross-Validation Summary
    print("\n" + "="*50)
    print("CROSS-VALIDATION COMPLETE")
    print(f"Hyperparameters: Batch Size = {args.batch_size}, LR = {args.learning_rate}, Dropout = {args.dropout_rate}, Weight Decay = {args.weight_decay}")
    print("="*50)

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    print(f"Average Validation Accuracy over {kfold.n_splits} folds: {mean_accuracy:.2f}%")
    print(f"Standard Deviation: {std_accuracy:.2f}%")

    # Save results to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok = True)

    # Write header only if file doesn't exist
    if not os.path.exists(args.output_file):
        with open(args.output_file, "w") as f:
            f.write("batch_size,epochs,learning_rate,weight_decay,dropout_rate,mean_accuracy,std_accuracy\n")
    
    with open(args.output_file, "a") as f:
        f.write(f"{args.batch_size},{args.epochs},{args.learning_rate},{args.weight_decay},{args.dropout_rate},{mean_accuracy:.2f}%,{std_accuracy:.2f}%\n")

    print(f"Results saved to {args.output_file}")
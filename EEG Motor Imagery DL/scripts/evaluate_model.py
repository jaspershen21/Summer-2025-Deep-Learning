import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import sys

# Add "scripts" directory to the Python path
module_path = os.path.abspath(os.path.join(os.path.join("..")))
if module_path not in sys.path:
    sys.path.append(module_path)

from scripts.dataset import BCIDataset
from scripts.model import EEGNet

PROCESSED_DATA_DIR = "./../data/processed/"
PLOT_DIR = "./../results/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data and Model
evaluation_dataset = BCIDataset(PROCESSED_DATA_DIR, dataset_type = "E")
evaluation_dataloader = DataLoader(evaluation_dataset, batch_size = 16, shuffle = False)

model = EEGNet()
model.load_state_dict(torch.load("./../models/best_model.pth")["model_state_dict"])
model.to(device)

# Gather Predictions and True Labels
model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for inputs, labels in evaluation_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.numpy())

accuracy = accuracy_score(true_labels, predictions) * 100
class_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]

print("\n--- Evaluation Results ---")
print(f"Accuracy: {accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=class_names))

cm = confusion_matrix(true_labels, predictions)

display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)
display.plot()
plot_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
plt.savefig(plot_path)
plt.show()
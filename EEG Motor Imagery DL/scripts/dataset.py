import mne
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class BCIDataset(Dataset):
    def __init__(self, data_path, dataset_type = "T"):
        """
        Args:
            data_path (str): Path to the directory containing the processed EEG data files.
            dataset_type (str): Type of dataset, either "E" for evaluation or "T" for training.
        """
        self.data_path = data_path
        self.dataset_type = dataset_type
        
        # Load epochs as list of values and their labels
        file_names = [f for f in sorted(os.listdir(data_path)) if dataset_type in f]
        all_epochs = [os.path.join(data_path, f) for f in file_names]
        combined_epoch = mne.concatenate_epochs([mne.read_epochs(f, preload = True) for f in all_epochs])

        self.X = combined_epoch.get_data(picks = "eeg", copy = False).astype(np.float32)
        self.X = np.expand_dims(self.X, axis = 1) # Add channel dimension (batch_size, 1, channels, time)
        first_event_id = 769
        self.y = (combined_epoch.events[:, -1] - first_event_id).astype(np.int32) # Zero-index labels

        print(f"Loaded a total of {len(self.X)} trials.")
        print(f"Data shape (X): {self.X.shape}") # Should be (trials, 1, channels, time)
        print(f"Labels shape (y): {self.y.shape}") # Should be (trials,)

    def __len__(self):
        """ Returns the number of samples in the dataset. """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Returns a single sample and its label at the specified index.

        Returns:
            (torch.Tensor, torch.Tensor): A tuple containing the EEG sample and its corresponding label.
        """

        sample = torch.from_numpy(self.X[idx])
        label = torch.from_numpy(np.array(self.y[idx]))

        return sample, label # (X, y) pair
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ClinicalDataset(Dataset):
    def __init__(self, structured_data, text_data, labels):
        self.structured_data = structured_data  # (num_samples, K features)
        self.text_data = text_data  # List of tensors with shape (N_i, 1024)
        self.labels = labels  # Dictionary with 'diagnosis', 'intervention', etc.
        
    def __len__(self):
        return len(self.structured_data)
    
    def __getitem__(self, idx):
        structured_input = self.structured_data[idx]
        text_input = self.text_data[idx]
        label = {key: self.labels[key][idx] for key in self.labels}
        return structured_input, text_input, label

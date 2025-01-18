import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredDataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StructuredDataEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer for structured data
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        # x: (batch_size, K)
        # Mask NaN values
        mask = ~torch.isnan(x)
        x = torch.where(mask, x, torch.zeros_like(x))
        
        # Apply embedding
        embedded = self.embedding(x)
        # Apply mask to the embeddings
        embedded = embedded * mask.unsqueeze(-1).float()
        
        return embedded  # (batch_size, hidden_dim)

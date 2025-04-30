import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv


class GCNnet(nn.Module):
    def __init__(self, num_features = 40 , n_output = 1, output_dim = 128, dropout = 0.2):
        super(GCNnet, self).__init__()
        self.conv1 = GCNConv(num_features, num_features)
        self.conv2 = GCNConv(num_features, num_features*2)
        self.conv3 = GCNConv(num_features*2, num_features*4)
        
        self.relu = Relu()
        self.fc1 = Linear(num_features*4, output_dim)
        self.fc2 = Linear(output_dim, n_output)
        
        self.dropout = nn.Dropout(dropout)
        self.out = F.softmax() 
        
    def forward(self, data):

        return out

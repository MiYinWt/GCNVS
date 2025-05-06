import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout, Softmax
from torch_geometric.nn import GCNConv


class GCNnet(nn.Module):
    def __init__(self, num_features=40, n_output=1, output_dim=128, dropout= 0.2):
        super(GCNnet, self).__init__()
        self.conv1 = GCNConv(num_features, num_features)
        self.conv2 = GCNConv(num_features, num_features*2)
        self.conv3 = GCNConv(num_features*2, num_features*4)
        
        self.relu = ReLU()
        self.fc1 = Linear(num_features*4, output_dim)
        self.fc2 = Linear(output_dim, n_output)
        
        self.dropout = Dropout(dropout)

        
    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index ,data.edge_weights

        x = self.conv1(x, edge_index, edge_weights)
        x = self.relu(x)

        x = self.conv2(x, edge_index, edge_weights)
        x = self.relu(x)

        x = self.conv3(x, edge_index, edge_weights)
        x = self.relu(x)
        x = torch.mean(x, dim=0, keepdim=True)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        out = Softmax(x,dim=1)

        return out

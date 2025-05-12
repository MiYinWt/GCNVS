import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool as gmp


class GCNnet(nn.Module):
    def __init__(self, num_features=40, output_dim=128, dropout= 0.2):
        super(GCNnet, self).__init__()
        self.conv1 = GCNConv(num_features, num_features)
        self.conv2 = GCNConv(num_features, num_features*2)
        self.conv3 = GCNConv(num_features*2, output_dim)
        
        self.relu = ReLU()
        self.fc1 = Linear(output_dim, 512)
        self.fc2 = Linear(512, 2)
        
        self.dropout = Dropout(dropout)

        
    def forward(self, data):
        x, edge_index, edge_weights,batch = data.x, data.edge_index ,data.edge_weights,data.batch

        x = self.conv1(x, edge_index, edge_weights)
        x = self.relu(x)

        x = self.conv2(x, edge_index, edge_weights)
        x = self.relu(x)

        x = self.conv3(x, edge_index, edge_weights)
        x = self.relu(x)
        
        x = gmp(x, batch)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        out = F.softmax(x,dim=1)

        return out

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool as gmp


class GCNnet(nn.Module):
    def __init__(self, num_features=26, dropout= 0.5):
        super(GCNnet, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.relu = ReLU()
        self.fc1 = Linear(128, 64)
        self.fc2 = Linear(64, 2)
        self.dropout = Dropout(dropout)

        
    def forward(self, data):
        x, edge_index, edge_weights,batch = data.x, data.edge_index ,data.edge_weights,data.batch
        x = self.conv1(x, edge_index, edge_weights)
        x = self.conv2(x, edge_index, edge_weights)
        x = self.conv3(x, edge_index, edge_weights)

        x = self.relu(x)
        x = self.dropout(x)

        x = gmp(x, batch)
        x = self.fc1(x)
        
        x = self.relu(x)
        out = self.fc2(x)

        #out = F.softmax(x,dim=1)

        return out

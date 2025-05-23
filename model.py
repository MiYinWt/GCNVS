import torch
import torch.nn as nn
from torch.nn import Dropout,Linear, ReLU
from torch.nn import functional as F
from torch_geometric.nn import GCNConv,GATConv, global_max_pool as gmp


class GCNnet(nn.Module):
    def __init__(self, num_features=32, dropout= 0.7):
        super(GCNnet, self).__init__()

        self.conv1 = GATConv(num_features, num_features, heads=8)
        self.conv2 = GCNConv(num_features * 8, 512)
        self.conv3 = GCNConv(512, 512)

        self.res_fc = Linear(num_features * 8, 512)

        self.relu = ReLU()
        self.fc1 = Linear(512, 256)
        self.fc2 = Linear(256, 2)
        self.dropout = Dropout(dropout)

        self.bn1 = nn.BatchNorm1d(num_features * 8)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, data):
        x, edge_index, edge_weights,batch = data.x, data.edge_index ,data.edge_weights,data.batch
        #print("x:\n",x,edge_weights)
        x = self.conv1(x, edge_index, edge_weights)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x, edge_index, edge_weights)
        x = self.bn2(x)
        x = self.relu(x)


        x = self.conv3(x, edge_index, edge_weights)
        x = self.bn3(x)
        x = self.relu(x)

        x = gmp(x, batch)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)

        # out = F.softmax(x,dim=1)

        return out
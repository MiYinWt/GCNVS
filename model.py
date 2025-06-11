import torch
import torch.nn as nn
from torch.nn import Dropout,Linear, ReLU
from torch.nn import functional as F
from torch_geometric.nn import GCNConv,GATConv, global_max_pool as gmp, global_mean_pool as gap,Set2Set


class GCNnet(nn.Module):
    def __init__(self, num_features=32, dropout= 0.5):
        super(GCNnet, self).__init__()

        self.conv1 = GATConv(num_features, num_features,heads=8,dropout=0.3)
    
        self.conv2 = GCNConv(num_features * 8, 256)
        self.conv3 = GCNConv(256, 1024)
        self.relu = ReLU()

        self.global_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            Dropout(dropout),
        )

        self.fc1 = Linear(1024, 256)
        self.fc2 = Linear(256, 2)
        self.dropout = Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, data):
        x, edge_index, edge_weights,batch,graph_features = data.x, data.edge_index ,data.edge_weights,data.batch,data.graph_features
        
        graph_features = graph_features.view(-1, 2048)
        g = self.global_fc(graph_features)
        #print("x:\n",x,edge_weights)
        x = self.conv1(x, edge_index, edge_weights)
        x = self.relu(x)

        x = self.conv2(x, edge_index, edge_weights)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv3(x, edge_index, edge_weights)
        x = self.bn2(x)
        x = self.relu(x)

        #x = x + g[batch]
        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = gmp(x, batch)
    
        x = x + g
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        out = self.fc2(x)

        # out = F.softmax(x,dim=1)

        return out

class CNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=8, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, 2)

    def forward(self, data):
        graph_features = data.graph_features.view(-1, 1, 2048)  # [batch, channel=1, length]
        x = self.conv1(graph_features)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.fc(x)
        return out
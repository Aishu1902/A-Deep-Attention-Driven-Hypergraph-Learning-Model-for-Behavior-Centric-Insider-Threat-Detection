import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dropout_adj

class ResHybNet(nn.Module):
    def __init__(self, input_dim, output_dim, cnn='', gnn='', residual=''):
        super(ResHybNet, self).__init__()

        # Mode selection
        if cnn == '':
            self.mode = 'gnn'
        elif gnn == '':
            self.mode = 'cnn'
        else:
            self.mode = 'hybrid'

        # CNN Layers
        if cnn == 'CNN':
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

        # GNN Layers
        if gnn == 'GCN':
            self.gnn1 = GCNConv(input_dim, 32)
            self.gnn2 = GCNConv(32, output_dim)

        elif gnn == 'SAGE':
            self.gnn1 = SAGEConv(input_dim, 32)
            self.gnn2 = SAGEConv(32, output_dim)

        elif gnn == 'GAT':
            self.gnn1 = GATConv(input_dim, 8, heads=4, concat=True, dropout=0.6)   # output: 32
            self.gnn2 = GATConv(32, 8, heads=4, concat=True, dropout=0.6)          # output: 32
            self.gnn3 = GATConv(32, 16, heads=2, concat=True, dropout=0.6)         # output: 32
            self.gnn4 = GATConv(32, output_dim, heads=1, concat=False, dropout=0.6)

        # Output layers
        self.output = None
        self.output_g = nn.Linear(output_dim, 2)  # output_dim comes from GNN layer

        # Dummy input to compute CNN output size dynamically
        if cnn == 'CNN' or self.mode == 'hybrid':
            dummy_input = torch.zeros(1, 1, input_dim)
            dummy_out = self.conv1(dummy_input)
            dummy_out = self.conv2(dummy_out)
            flattened_size = dummy_out.view(1, -1).size(1)
            self.output = nn.Linear(flattened_size, 2)

        # Residual handling
        self.res = True if residual == 'YES' else False
        if self.res:
            self.res_scale = nn.Parameter(torch.tensor(0.5))  # learnable residual weight

        self.layer_norm_x = nn.LayerNorm(input_dim)
        self.layer_norm_g = nn.LayerNorm(output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.mode == 'cnn':
            x_c = torch.unsqueeze(x, dim=1)
            x_c = self.conv1(x_c)
            x_c = self.conv2(x_c)
            x_c = x_c.view(x_c.size(0), -1)
            out = self.output(x_c)

        elif self.mode == 'gnn':
            if hasattr(self, 'gnn4'):  # deeper GAT
                x_g = self.gnn1(x, edge_index)
                x_g = F.elu(x_g)
                x_g = F.dropout(x_g, p=0.3, training=self.training)

                x_g = self.gnn2(x_g, edge_index)
                x_g = F.elu(x_g)
                x_g = F.dropout(x_g, p=0.3, training=self.training)

                x_g = self.gnn3(x_g, edge_index)
                x_g = F.elu(x_g)
                x_g = F.dropout(x_g, p=0.3, training=self.training)

                x_g = self.gnn4(x_g, edge_index)
            else:
                x_g = self.gnn1(x, edge_index)
                x_g = F.elu(x_g)
                x_g = F.dropout(x_g, p=0.3, training=self.training)
                x_g = self.gnn2(x_g, edge_index)

            out = self.output_g(x_g)

        else:  # hybrid
            if hasattr(self, 'gnn4'):  # deeper GAT
                x_g = self.gnn1(x, edge_index)
                x_g = F.elu(x_g)
                x_g = F.dropout(x_g, p=0.3, training=self.training)

                x_g = self.gnn2(x_g, edge_index)
                x_g = F.elu(x_g)
                x_g = F.dropout(x_g, p=0.3, training=self.training)

                x_g = self.gnn3(x_g, edge_index)
                x_g = F.elu(x_g)
                x_g = F.dropout(x_g, p=0.3, training=self.training)

                x_g = self.gnn4(x_g, edge_index)
            else:
                x_g = self.gnn1(x, edge_index)
                x_g = F.elu(x_g)
                x_g = F.dropout(x_g, p=0.3, training=self.training)
                x_g = self.gnn2(x_g, edge_index)

            if self.res:
                x_norm = self.layer_norm_x(x)
                xg_norm = self.layer_norm_g(x_g)

                if x_norm.shape[1] != xg_norm.shape[1]:
                    self.res_proj = nn.Linear(xg_norm.size(1), x_norm.size(1)).to(x.device)
                    xg_norm = self.res_proj(xg_norm)

                x_dual = x_norm + self.res_scale * xg_norm
            else:
                x_dual = x_g

            x_dual = torch.unsqueeze(x_dual, dim=1)
            x_dual = self.conv1(x_dual)
            x_dual = self.conv2(x_dual)
            x_dual = x_dual.view(x_dual.size(0), -1)
            out = self.output(x_dual)

        return F.log_softmax(out, dim=1)

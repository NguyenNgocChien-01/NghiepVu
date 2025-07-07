import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv

class InvoiceGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes,
                 dropout_rate=0.5, chebnet=False, K=3):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.chebnet = chebnet
        self.K = K

        # Sử dụng ModuleList để chứa các lớp ẩn một cách linh hoạt
        self.convs = nn.ModuleList()

        # Xác định loại lớp Conv sẽ sử dụng
        ConvLayer = ChebConv if self.chebnet else GCNConv

        # --- Tự động tạo các lớp dựa trên hidden_dims ---

        # Lớp đầu tiên: từ input_dim -> hidden_dims[0]
        in_channels = input_dim
        out_channels = hidden_dims[0]
        if self.chebnet:
            self.convs.append(ConvLayer(in_channels, out_channels, K=self.K))
        else:
            self.convs.append(ConvLayer(in_channels, out_channels, improved=True, cached=True))

        # Các lớp ẩn tiếp theo: từ hidden_dims[i] -> hidden_dims[i+1]
        for i in range(len(hidden_dims) - 1):
            in_channels = hidden_dims[i]
            out_channels = hidden_dims[i+1]
            if self.chebnet:
                self.convs.append(ConvLayer(in_channels, out_channels, K=self.K))
            else:
                self.convs.append(ConvLayer(in_channels, out_channels, improved=True, cached=True))

        # Lớp cuối cùng (đầu ra): từ hidden_dims[-1] -> n_classes
        self.classifier_head = ConvLayer(hidden_dims[-1], n_classes,
                                         K=self.K if self.chebnet else None)
        if not self.chebnet:
             # Gán các tham số cho GCNConv nếu cần
            self.classifier_head.improved = True
            self.classifier_head.cached = True


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # Truyền qua các lớp ẩn trong ModuleList
        for conv_layer in self.convs:
            x = conv_layer(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Truyền qua lớp phân loại cuối cùng
        x = self.classifier_head(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)
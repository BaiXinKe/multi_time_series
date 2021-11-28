import torch
import torch.nn as nn
import numpy as np


class TCNLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel=3, dropout=0.5):
        super(TCNLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.conv2 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.conv3 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.bn = nn.BatchNorm2d(out_features)
        self.dropout = dropout

    def forward(self, inputs):
        """
        param inputs: (batch_size, timestamp, num_node, in_features)
        return: (batch_size, timestamp - 2, num_node, out_features)
        """
        inputs = inputs.permute(0, 3, 2, 1)  # (btnf->bfnt)
        out = self.conv1(inputs) + torch.sigmoid(self.conv2(inputs))
        out = torch.relu(out + self.conv3(inputs))
        out = self.bn(out)
        out = out.permute(0, 3, 2, 1)
        out = torch.dropout(out, p=self.dropout, train=self.training)
        return out


class TCN(nn.Module):
    def __init__(self, n_history, in_features, mid_features) -> None:
        super(TCN, self).__init__()
        # odd time seriers: number of layer is n_hitory // 2
        # even time seriers: number of layer is n_history//2-1 + a single conv layer.
        # -> Aggregate information from time seriers to one unit
        assert(n_history >= 3)
        self.is_even = False if n_history % 2 != 0 else True

        self.n_layers = n_history // \
            2 if n_history % 2 != 0 else (n_history // 2 - 1)

        self.tcn_layers = nn.ModuleList([TCNLayer(in_features, mid_features)])
        for i in range(self.n_layers - 1):
            self.tcn_layers.append(TCNLayer(mid_features, mid_features))

        if self.is_even:
            self.tcn_layers.append(
                TCNLayer(mid_features, mid_features, kernel=2))

        self.upsample = None if in_features == mid_features else nn.Linear(
            in_features, mid_features)

        self.reset_parameter()

    def reset_parameter(self):
        if self.upsample is not None:
            for param in self.upsample.parameters():
                param.data.normal_()

    def forward(self, inputs):
        out = self.tcn_layers[0](inputs)
        if self.upsample:
            inputs = self.upsample(inputs)

        out = out + inputs[:, 2:, ...]

        for i in range(1, self.n_layers):
            out = self.tcn_layers[i](out) + out[:, 2:, ...]

        if self.is_even:
            out = self.tcn_layers[-1](out) + out[:, -1, :, :].unsqueeze(1)

        return out


class Encoder(nn.Module):
    def __init__(self, n_history, n_predict, in_features, mid_features) -> None:
        super(Encoder, self).__init__()
        assert(n_history >= 3)
        self.n_predict = n_predict

        self.tcn = TCN(n_history=n_history,
                       in_features=in_features, mid_features=mid_features)

        self.fully = nn.Linear(mid_features, n_predict * mid_features)

        self.reset_parameter()

    def reset_parameter(self):
        for param in self.fully.parameters():
            param.data.normal_()

    def forward(self, inputs):
        out = self.tcn(inputs)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = torch.relu(self.fully(out))
        out = out.reshape(out.shape[0], out.shape[1], self.n_predict, -1)
        out = out.permute(0, 2, 1, 3)
        return out


class GCNCell(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5) -> None:
        super(GCNCell, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.bais = nn.Parameter(torch.FloatTensor(out_features))

        self.dropout = dropout

        self.reset_parameter()

    def reset_parameter(self):
        self.weight.data.normal_()
        self.bais.data.zero_()

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, timestamp, num_node, num_features)
        adj: (num_node, num_node)
        """
        lfs = torch.einsum('ij,jbtf->bitf', adj, inputs.permute(2, 0, 1, 3))
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bais)
        return result.permute(0, 2, 1, 3)


class Archer(nn.Module):
    def __init__(self, n_history, n_predict, in_features, mid_features, out_features, adj) -> None:
        super(Archer, self).__init__()
        self.adj = nn.Parameter(torch.from_numpy(adj))
        self.encoder = Encoder(n_history, n_predict,
                               in_features, mid_features)
        self.gcn = GCNCell(mid_features, mid_features)
        self.fully = nn.Linear(mid_features, mid_features)
        self.predict = nn.Linear(mid_features, out_features)

    def forward(self, inputs):
        out = self.encoder(inputs)
        out = self.gcn(out, self.adj)
        out = torch.relu(self.fully(out))
        out = self.predict(out)
        return out


if __name__ == '__main__':
    adj = np.random.randn(307, 307).astype(np.float32)
    data = torch.randn(64, 12, 307, 1)
    net = Archer(12, 12, 1, 32, 1, adj)
    total = 0
    for param in net.parameters():
        total += param.numel()
    print(total)

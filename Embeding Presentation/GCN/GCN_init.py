import torch
import numpy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary


def get_degree(A, symmetric=True):
    A = A + torch.eye(A.shape[0])
    d = A.sum(dim=-1)
    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))
        return torch.matmul(torch.matmul(D, A), D)
    else:
        D = torch.diag(torch.pow(d, -1))
        return torch.matmul(D, A)


def mask_select(y, mask):
    num = int(sum(mask))
    new_y = torch.zeros(num, y.shape[1])
    k = 0
    for i in range(mask.shape[0]):
        if mask[i]:
            new_y[k] = y[i]
            k += 1
    return new_y


class GCN(nn.Module):
    def __init__(self, A, in_dim, out_dim):
        super(GCN, self).__init__()
        self.A = get_degree(A)
        self.in_dim = in_dim
        self.out_dim = out_dim
        hidden1 = self.in_dim//2
        hidden2 = hidden1 // 2
        self.fc1 = nn.Linear(self.in_dim, hidden1)
        self.activate1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.activate2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden2, self.out_dim)
        self.activate3 = nn.Tanh()

    def forward(self, x):
        x = torch.matmul(self.A, x)
        x = self.activate1(self.fc1(x))
        x = torch.matmul(self.A, x)
        x = self.activate2(self.fc2(x))
        x = torch.matmul(self.A, x)
        x = self.activate3(self.fc3(x))
        return x

    def initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data, gain=1)
                # nn.init.xavier_normal_(layer.weight.data, gain=1)


if __name__ == '__main__':
    lr = 1e-2
    epochs = 100
    A = torch.ones(100, 100)
    model = GCN(A, 1000, 768)
    model.initialize()
    x = torch.randn(100, 1000)
    opt = optim.Adam(model.parameters(), lr=lr)

    output = model(x).detach()
    print(output)
    print(output.requires_grad)
    summary(model, (100, 1000))













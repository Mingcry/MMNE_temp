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
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, self.in_dim//2)
        self.dropout1 = nn.Dropout(0.1)
        self.activate1 = nn.ReLU()
        self.fc2 = nn.Linear(self.in_dim//2, self.out_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.activate2 = nn.ReLU()
        self.fc3 = nn.Linear(self.out_dim, 1)

    def forward(self, A, x):
        A = get_degree(A)
        x = torch.matmul(A, x)
        x = self.activate1(self.dropout1(self.fc1(x)))
        x = torch.matmul(A, x)
        # x = self.activate2(self.dropout2(self.fc2(x)))
        return self.fc2(x)


if __name__ == '__main__':
    lr = 1e-2
    epochs = 500
    A = torch.ones(100, 100)
    model = GCN(256, 2)
    opt = optim.Adam(model.parameters(), lr=lr)

    label = torch.zeros(100).long()
    label[0] = 1
    label[33] = 1
    label[10] = 1
    label[79] = 1
    label[23] = 0
    label[67] = 0

    label_mask = torch.zeros(100)
    label_mask[0] = 1
    label_mask[33] = 1
    label_mask[10] = 1
    label_mask[79] = 1
    label_mask[23] = 1
    label_mask[67] = 1

    x = torch.randn(100, 256)

    for epoch in range(epochs):
        pred = F.softmax(model(A, x), dim=-1)
        loss = (-pred.log().gather(1, label.view(-1, 1)))
        loss = mask_select(loss, label_mask).mean()
        if epoch % 20 == 0 and epoch != 0:
            print('[{}/{}]'.format(epoch, epochs), 'loss={:.3f}'.format(loss.item()))

        opt.zero_grad()
        loss.backward()
        opt.step()






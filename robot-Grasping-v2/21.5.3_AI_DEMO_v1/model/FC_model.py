import torch

class FC(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(FC, self).__init__()
        self.linear = torch.nn.Linear(in_size, out_size)

        fc1 = torch.nn.Linear(in_size, 128)
        fc2 = torch.nn.Linear(128, 64)
        fc3 = torch.nn.Linear(64, 32)
        fc4 = torch.nn.Linear(32, out_size)
        dropout = torch.nn.Dropout(p=0.25)

        self.fc_module = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            dropout,
            fc2,
            torch.nn.ReLU(),
            dropout,
            fc3,
            torch.nn.ReLU(),
            dropout,
            fc4
        )
        #
        # self.fc_module = torch.nn.Sequential(
        #     fc1,
        #     torch.nn.ReLU(),
        #     fc2,
        #     torch.nn.ReLU(),
        #     fc3,
        #     torch.nn.ReLU(),
        #     fc4
        # )

    def forward(self, x):
        out = self.fc_module(x)
        return out


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class MLP_Model(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
        loss_func,
        in_dim: int = 8,
        n_layers: int = 2,
        hid_dim: int = 90,
        lr: float = 1e-3,
    ):
        super(MLP_Model, self).__init__()

        self.hidden_layers = n_layers

        self.loss_func = loss_func

        match (self.hidden_layers):
            case 1:
                self.fc1 = nn.Linear(in_dim, hid_dim)
            case 2:
                self.fc1 = nn.Linear(in_dim, hid_dim)
                self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.output = nn.Linear(hid_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.num_params = nn.utils.parameters_to_vector(self.parameters()).size()[0]

    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, params: torch.Tensor):
        nn.utils.vector_to_parameters(params, self.parameters())

    def forward(self, x: torch.Tensor):
        out = F.tanh(self.fc1(x))

        match (self.hidden_layers):
            case 2:
                out = F.tanh(self.fc2(out))

        return self.output(out)

    def train(self, x: torch.Tensor, y: torch.Tensor):

        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLP_Model(loss_func=2).to(device)

    print(model.num_params)

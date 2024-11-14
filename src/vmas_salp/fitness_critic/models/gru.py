import torch
import torch.optim as optim
import torch.nn as nn


class GRU_Model(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
        loss_func,
        n_layers: int = 1,
        in_dim: int = 8,
        hid_dim: int = 50,
        lr: float = 1e-3,
    ):
        super(GRU_Model, self).__init__()

        self.loss_func = loss_func

        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True)
        self.output = nn.Linear(hid_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.num_params = nn.utils.parameters_to_vector(self.parameters()).size()[0]

    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, params: torch.Tensor):
        nn.utils.vector_to_parameters(params, self.parameters())

    def forward(self, x: torch.Tensor):

        self.rnn.flatten_parameters()
        _, hn = self.rnn(x)
        return torch.squeeze(self.output(hn), dim=0)

    def train(self, x: torch.Tensor, y: torch.Tensor):

        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GRU_Model(loss_func=2).to(device)

    print(model.num_params)

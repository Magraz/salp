import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np


class GRU_Policy(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
        input_size: int,
        n_layers: int,
        hidden_size: int,
        output_size: int,
    ):
        super(GRU_Policy, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_size, n_layers, dtype=torch.double)
        self.output = nn.Linear(hidden_size, output_size, dtype=torch.double)
        self.num_params = nn.utils.parameters_to_vector(self.parameters()).size()[0]

        # Disable gradient calcs
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        self.rnn.flatten_parameters()
        _, hn = self.rnn(x.unsqueeze(0))
        out = self.output(hn)
        return F.tanh(out)

    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, params: torch.Tensor):
        nn.utils.vector_to_parameters(params, self.parameters())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GRU_Policy(input_size=8, hidden_size=17, output_size=2, n_layers=1).to(
        device
    )
    model_copy = deepcopy(model)

    torch.set_printoptions(threshold=10_000)
    print(model_copy.num_params)

    # print(model_copy.get_params())

    input = torch.tensor([[-1, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.double).to(
        device
    )
    print(model_copy.forward(input))

    rand_params = torch.rand(model_copy.get_params().size()).to(device)
    mutated_params = torch.add(model_copy.get_params(), rand_params).to(device)

    model_copy.set_params(mutated_params)

    print(model_copy.forward(input))

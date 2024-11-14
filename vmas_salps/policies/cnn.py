import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class CNN_Policy(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
        img_size,
    ):
        super(CNN_Policy, self).__init__()

        self.cnn = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=img_size // 4)
        self.maxpool = nn.AvgPool2d(kernel_size=img_size // 6, stride=2)
        self.linear = nn.Linear(648, 2)
        self.num_params = nn.utils.parameters_to_vector(self.parameters()).size()[0]

    def forward(self, x: torch.Tensor):
        out = self.cnn(x)
        out = F.leaky_relu(out)
        out = self.maxpool(out)
        out = self.linear(out.flatten())
        return F.tanh(out)

    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, params: torch.Tensor):
        nn.utils.vector_to_parameters(params, self.parameters())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN_Policy(img_size=25).to(device)
    model_copy = deepcopy(model)
    print(model_copy.num_params)

    input = torch.ones((2, 25, 25), dtype=torch.float).to(device)
    print(model_copy.forward(input))

    rand_params = torch.rand(model_copy.get_params().size()).to(device)
    mutated_params = torch.add(model_copy.get_params(), rand_params).to(device)

    model_copy.set_params(mutated_params)

    print(model_copy.forward(input))

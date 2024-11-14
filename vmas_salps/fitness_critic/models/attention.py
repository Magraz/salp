import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, device: str
) -> torch.Tensor:

    # Efficient implementation equivalent to the following:
    L, D = query.size(-2), key.size(-1)
    scale_factor = 1 / math.sqrt(D)
    attn_bias = torch.zeros(L, L, dtype=query.dtype).to(device)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value, attn_weight


def getPositionEncoding(seq_len: int, d: int, n: int = 10000):
    P = np.zeros((seq_len, d))

    for k in range(seq_len):

        for i in np.arange(int(d / 2)):

            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)

    return P


class Attention_Model(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
        loss_func,
        device: str,
        seq_len: int,
        in_dim: int = 8,
        hid_dim: int = 90,
        lr: float = 1e-3,
    ):
        super(Attention_Model, self).__init__()

        self.device = device

        self.loss_func = loss_func

        self.w1 = nn.Linear(in_dim, hid_dim)

        self.w_out1 = nn.Linear(hid_dim, hid_dim)
        self.w_out2 = nn.Linear(hid_dim, 1)
        self.w_out3 = nn.Linear(seq_len, 1)

        self.double()
        self.to(device)

        self.pos_enc = getPositionEncoding(seq_len, in_dim)
        self.pos_enc = torch.from_numpy(self.pos_enc.astype(np.float64)).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.num_params = nn.utils.parameters_to_vector(self.parameters()).size()[0]

    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, params: torch.Tensor):
        nn.utils.vector_to_parameters(params, self.parameters())

    def forward(self, x: torch.Tensor):

        KQV = self.w1(x + self.pos_enc)
        res = F.scaled_dot_product_attention(KQV, KQV, KQV)
        transformout = self.w_out2(F.leaky_relu(self.w_out1(res)))
        transformout = torch.flatten(transformout, start_dim=-2, end_dim=-1)

        return self.w_out3(transformout)

    def train(self, x: torch.Tensor, y: torch.Tensor, shaping=False):

        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Attention_Model(loss_func=2, device=device, seq_len=31)

    print(model.num_params)

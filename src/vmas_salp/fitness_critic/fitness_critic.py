from collections import deque
import numpy as np
from random import sample
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from vmas_rovers.fitness_critic.models.mlp import MLP_Model
from vmas_rovers.fitness_critic.models.attention import Attention_Model
from vmas_rovers.fitness_critic.models.gru import GRU_Model
from vmas_rovers.learning.ccea.types import FitnessCriticType
from vmas_rovers.utils.loss_functions import alignment_loss


class TrajectoryRewardDataset(Dataset):

    def __init__(self, traj_hist, model_type: str):

        if len(traj_hist) < 256:
            trajG = traj_hist
        else:
            trajG = sample(traj_hist, 256)

        self.observations, self.reward = [], []

        for traj, g in trajG:

            match model_type:

                case FitnessCriticType.MLP:
                    for s in traj:  # train whole trajectory
                        self.observations.append(s.tolist())
                        self.reward.append([g])

                case FitnessCriticType.ATT | FitnessCriticType.GRU:
                    self.observations.append(traj.tolist())
                    self.reward.append([g])

        self.observations, self.reward = np.array(
            self.observations, dtype=np.float32
        ), np.array(self.reward, dtype=np.float32)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):

        return (
            self.observations[idx],
            self.reward[idx],
        )


class FitnessCritic:
    def __init__(
        self,
        device: str,
        model_type: str,
        loss_fn: int,
        episode_size: int,
        hidden_size: int,
        n_layers: int,
    ):

        self.hist = deque(maxlen=30000)
        self.device = device

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Set loss function
        if loss_fn == 0:
            self.loss_func = nn.MSELoss(reduction="sum")
        elif loss_fn == 1:
            self.loss_func = alignment_loss
        elif loss_fn == 2:
            self.loss_func = lambda x, y: alignment_loss(x, y) + nn.MSELoss(
                reduction="sum"
            )(x, y)

        # Set model type
        match self.model_type:
            case FitnessCriticType.MLP:
                self.model = MLP_Model(
                    loss_func=self.loss_func,
                    hid_dim=self.hidden_size,
                    n_layers=self.n_layers,
                ).to(device)
                self.batch_size = episode_size + 1

            case FitnessCriticType.ATT:
                self.model = Attention_Model(
                    loss_func=self.loss_func, device=device, seq_len=episode_size + 1
                )
                self.batch_size = 1

            case FitnessCriticType.GRU:
                self.model = GRU_Model(
                    loss_func=self.loss_func,
                    hid_dim=self.hidden_size,
                    n_layers=self.n_layers,
                ).to(device)
                self.batch_size = 1

        self.params = self.model.get_params()

    def add(self, trajectory, G):
        self.hist.append((trajectory, G))

    def evaluate(self, trajectory):  # evaluate max state
        result = self.model.forward(trajectory).cpu().detach().numpy()
        return np.max(result)

    def train(self, epochs: int):

        avg_loss = []

        traj_dataset = TrajectoryRewardDataset(self.hist, self.model_type)

        for _ in range(epochs):

            accum_loss = 0
            batches = 0

            dataloader = DataLoader(
                traj_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

            for x, y in dataloader:
                accum_loss += self.model.train(x.to(self.device), y.to(self.device))
                batches += 1

            avg_loss.append(accum_loss / batches)

        return np.mean(np.array(avg_loss))

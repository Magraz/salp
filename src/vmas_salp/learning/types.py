from dataclasses import dataclass


class Team:
    def __init__(
        self,
        idx: int,
        individuals: list = None,
        combination: list = None,
    ):
        self.idx = idx
        self.individuals = individuals if individuals is not None else []
        self.combination = combination if combination is not None else []


class JointTrajectory:
    def __init__(self, joint_state_traj: list, joint_obs_traj: list):
        self.states = joint_state_traj
        self.observations = joint_obs_traj

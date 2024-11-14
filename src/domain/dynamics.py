from vmas.simulator.dynamics.common import Dynamics


class SalpDynamics(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 3

    def process_action(self):
        self.agent.state.force = self.agent.action.u[:, :2]
        self.agent.state.join = self.agent.action.u[:, 2]

    def check_and_process_action(self):
        action = self.agent.action.u
        if action.shape[1] < self.needed_action_size:
            raise ValueError(
                f"Agent action size {action.shape[1]} is less than the required dynamics action size {self.needed_action_size}"
            )
        self.process_action()

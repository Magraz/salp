from vmas.simulator.dynamics.common import Dynamics


class SalpDynamics(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self):
        self.agent.state.force = self.agent.action.u[:, :2]
        # self.agent.state.join = self.agent.action.u[:, 2]

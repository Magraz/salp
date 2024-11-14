from pynput.keyboard import Key, Listener
import numpy as np


class manual_control:
    def __init__(self, n_agents):
        self.controlled_agent = 0
        self.n_agents = n_agents
        self.command = [0, 0]
        self.speed = 0.25

    def on_press(self, key):

        try:
            match (key.char):
                case "w":
                    self.command = [0.0, self.speed]
                case "a":
                    self.command = [-self.speed, 0.0]
                case "s":
                    self.command = [0.0, -self.speed]
                case "d":
                    self.command = [self.speed, 0.0]

        except AttributeError:
            pass

    def on_release(self, key):

        self.command = [0.0, 0.0]

        if key == Key.space:
            self.controlled_agent += 1
            if self.controlled_agent == self.n_agents:
                self.controlled_agent = 0


if __name__ == "__main__":
    mc = manual_control(n_agents=1)
    # Collect events until released
    with Listener(on_press=mc.on_press, on_release=mc.on_release) as listener:
        listener.join()

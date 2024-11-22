from vmas.simulator.utils import Color
import random

COLOR_MAP = {
    "GREEN": Color.GREEN,
    "RED": Color.RED,
    "BLUE": Color.BLUE,
    "BLACK": Color.BLACK,
    "LIGHT_GREEN": Color.LIGHT_GREEN,
    "GRAY": Color.GRAY,
    "WHITE": Color.WHITE,
    "PURPLE": (0.75, 0.25, 0.75),
    "ORANGE": (0.75, 0.75, 0.25),
    "MAGENTA": (0.9, 0.25, 0.5),
}

def sample_filtered_normal(mean, std_dev, threshold):
    while True:
        # Sample a single value from the normal distribution
        value = random.normalvariate(mu=mean, sigma=std_dev)
        # Check if the value is outside the threshold range
        if abs(value) > threshold:
            return value

import numpy as np


class GridWorld:
    def __init__(self, location_to_state, actions, rewards):
        self.location_to_state = location_to_state
        self.actions = actions
        self.rewards = rewards
        self.state_to_location = dict(
            (state, location) for location, state in self.location_to_state.items()
        )

import numpy as np
from agent import QLearningAgent
from environment import GridWorld


if __name__ == "__main__":
    location_to_state = {
        "Thailand": 0,
        "China": 1,
        "Japan": 2,
        "USA": 3,
        "Canada": 4,
        "France": 5,
        "Germany": 6,
        "Italy": 7,
        "Greece": 8,
    }
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # rewards = np.random.randint(0, 2, (len(location_to_state), len(location_to_state)))
    rewards = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
        ]
    )
    env = GridWorld(location_to_state, actions, rewards)
    agent = QLearningAgent(env)

    agent.training("Thailand", "Greece", 1000)

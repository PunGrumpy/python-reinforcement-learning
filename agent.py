import random
import numpy as np


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.MAX_SIZE = len(self.env.location_to_state)
        self.q_table = np.zeros((self.MAX_SIZE, self.MAX_SIZE), dtype=None, order="C")

    def training(self, start_location, end_location, iterations):
        rewards_new = np.copy(self.env.rewards)

        ending_state = self.env.location_to_state[end_location]
        rewards_new[ending_state, ending_state] = 999

        # Picking a random current state
        for _ in range(iterations):
            current_state = np.random.randint(0, 9)
            playable_actions = []

            # Find all playable actions / rewards
            for j in range(self.MAX_SIZE):
                if rewards_new[current_state, j] > 0:
                    playable_actions.append(j)

            # Choose a random next state
            next_state = np.random.choice(playable_actions)

            # Find temporal difference
            TD = (
                rewards_new[current_state, next_state]
                + self.discount_factor
                * self.q_table[next_state, np.argmax(self.q_table[next_state,])]
                - self.q_table[current_state, next_state]
            )

            # Bellman equation: Q(s,a) = Q(s,a) + learning_rate * (reward + discount_factor * max(Q(s',a')) - Q(s,a))
            self.q_table[current_state, next_state] += self.learning_rate * TD

        route = [start_location]  # Define the route as the start location
        next_location = start_location  # Define the next location as the start location

        self.get_optimal_route(
            start_location, end_location, next_location, route, self.q_table
        )

    def get_optimal_route(self, start_location, end_location, next_location, route, Q):
        while next_location != end_location:
            start_state = self.env.location_to_state[start_location]
            # Find the next state by finding the max Q value
            next_state = np.argmax(Q[start_state,])
            next_location = self.env.state_to_location[next_state]
            route.append(next_location)
            start_location = next_location

        print(
            route.__str__()
            .replace(",", " ->")
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
        )

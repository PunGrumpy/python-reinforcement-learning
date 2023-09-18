import numpy as np


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize Q-table with zeros
        self.MAX_SIZE = len(self.env.location_to_state)
        self.q_table = np.zeros((self.MAX_SIZE, self.MAX_SIZE), dtype=float)

    def training(self, start_location, end_location, iterations):
        rewards_new = np.copy(self.env.rewards)

        # Set a high reward for reaching the end location
        ending_state = self.env.location_to_state[end_location]
        rewards_new[ending_state, ending_state] = 999

        for step in range(iterations):
            print(f"Step {step + 1}/{iterations}")

            # Choose a random current state
            current_state = np.random.randint(0, self.MAX_SIZE)
            print(f"Current State: {self.env.state_to_location[current_state]}")
            print(f"Q-Table:\n {self.q_table}\n")

            playable_actions = []

            # Find all playable actions (states with positive rewards)
            for j in range(self.MAX_SIZE):
                if rewards_new[current_state, j] > 0:
                    playable_actions.append(j)

            print(
                f"Playable Actions: {list(map(self.env.state_to_location.get, playable_actions))}"
            )

            # Choose a random next state from playable actions
            next_state = np.random.choice(playable_actions)
            print(f"Next State: {next_state}")

            # Calculate the Temporal Difference (TD) error
            TD = (
                rewards_new[current_state, next_state]
                + self.discount_factor * np.max(self.q_table[next_state, :])
                - self.q_table[current_state, next_state]
            )

            print(f"Temporal Difference (TD): {TD}")

            # Update Q-value using the Q-learning formula
            self.q_table[current_state, next_state] += self.learning_rate * TD

            print(f"Updated Q-Value: {self.q_table[current_state, next_state]}\n")

        # Find and print the optimal route
        optimal_route = self.find_optimal_route(start_location, end_location)
        print("Optimal Route:", optimal_route)

    def find_optimal_route(self, start_location, end_location):
        route = [start_location]
        current_location = start_location

        while current_location != end_location:
            current_state = self.env.location_to_state[current_location]

            # Get Q-values for the current state
            q_values = self.q_table[current_state, :]

            # Find the action (next state) with the highest Q-value
            next_state = np.argmax(q_values)

            # Explain the decision
            print(f"Current Location: {current_location}")
            print(
                f"Q-Values for Current State: {list(zip(self.env.state_to_location.values(), q_values.round(0)))}"
            )
            print(f"Choosing Next State: {next_state}")

            next_location = self.env.state_to_location[next_state]

            route.append(next_location)
            current_location = next_location

        return route

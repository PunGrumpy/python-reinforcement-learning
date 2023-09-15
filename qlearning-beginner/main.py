import gym
import random
import imageio
import numpy as np
from tqdm import trange


class QLearningAgent:
    def __init__(
        self,
        state_space,
        action_space,
        learning_rate=0.7,
        discount_rate=0.95,
        max_epsilon=1.0,
        min_epsilon=0.05,
        epsilon_decay_rate=0.0005,
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.QTable = np.zeros((state_space, action_space))

    def epsilon_greedy_policy(self, env, state, epsilon):
        # ทำการเลือก action จาก QTable โดยมีความน่าจะเป็น epsilon ที่จะเลือก action แบบสุ่ม
        # ถ้า random.uniform(0, 1) > epsilon จะเลือก action ที่ดีที่สุด แต่ถ้าน้อยกว่า epsilon จะเลือก action แบบสุ่ม
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(self.QTable[state])
        else:
            action = env.action_space.sample()  # สุ่ม action จาก env
        return action

    def train(self, env, episodes, max_steps):
        # ทำการเทรน agent ให้เรียนรู้จากการเล่นเกม
        for episode in trange(episodes):
            # หาค่า epsilon ที่จะใช้ในการเลือก action โดยใช้สูตร exponential decay
            # exponential decay คือ การลดค่า epsilon ลงเรื่อยๆ จนกว่าจะถึงค่า min_epsilon
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                -self.epsilon_decay_rate * episode
            )
            state = env.reset()

            for step in range(max_steps):
                # เอา state ปัจจุบันไปหา action ที่ดีที่สุด
                action = self.epsilon_greedy_policy(env, state, epsilon)
                # ทำการเล่นเกม 1 step จาก action ที่ได้จาก policy
                new_state, reward, done, info = env.step(action)
                # Step:
                # 0 = Left
                # 1 = Down
                # 2 = Right
                # 3 = Up

                # อัปเดตตาราง Q โดยใช้สมการ Bellman Optimality Equation
                # โดยใช้สูตร Q(s,a) = Q(s,a) + learning_rate * (reward + discount_rate * max(Q(s',a')) - Q(s,a))
                # โดย Q(s',a') คือค่า Q ที่ดีที่สุดของ state ใหม่
                self.QTable[state][action] = self.QTable[state][
                    action
                ] + self.learning_rate * (
                    reward
                    + self.discount_rate * np.max(self.QTable[new_state])
                    - self.QTable[state][action]
                )

                if done:
                    break

                state = new_state
        return self.QTable

    def evaluate_agent(self, env, max_steps, eval_episodes, QTable, rewards):
        # หาค่า reward ที่ได้จากการเทรน agent
        episode_rewards = []

        for episode in range(eval_episodes):
            # ถ้ามี rewards ให้ใช้ seed ในการ reset env
            if rewards:
                state = env.reset(seed=rewards[episode])
            else:
                state = env.reset()
            step = 0
            done = False
            total_rewards = 0

            for step in range(max_steps):
                # เอา state ปัจจุบันไปหา action ที่ดีที่สุด
                action = np.argmax(QTable[state][:])
                new_state, reward, done, info = env.step(action)
                # Step:
                # 0 = Left
                # 1 = Down
                # 2 = Right
                # 3 = Up
                total_rewards += reward

                if done:
                    break
                state = new_state
            episode_rewards.append(total_rewards)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward

    def record_video(self, env, out_directory, fps=1):
        # ทำการเล่นเกมและบันทึกเป็นวิดีโอ
        # วิธีการบันทึกวิดีโอนี้เป็นของ OpenAI Gym ที่มีอยู่แล้ว
        images = []
        done = False
        state = env.reset(seed=random.randint(0, 500))
        image = env.render(mode="rgb_array")
        images.append(image)

        while not done:
            # เอา state ปัจจุบันไปหา action ที่ดีที่สุด
            action = np.argmax(self.QTable[state][:])
            state, reward, done, info = env.step(action)
            image = env.render(mode="rgb_array")
            images.append(image)
        imageio.mimsave(
            out_directory, [np.array(img) for i, img in enumerate(images)], duration=fps
        )


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    state_space = env.observation_space.n
    action_space = env.action_space.n

    # Training Parameters
    episodes = 10000
    max_steps = 99

    # Evaluation Parameters
    eval_episodes = 100

    # Environment Parameters
    rewards = []

    agent = QLearningAgent(state_space, action_space)
    agent.train(env, episodes, max_steps)
    mean_reward, std_reward = agent.evaluate_agent(
        env, max_steps, eval_episodes, agent.QTable, rewards
    )
    print(agent.QTable)
    print(f"Mean Reward: {mean_reward} +/- {std_reward}")

    # บันทึกวิดีโอ
    agent.record_video(env, "frozenlake_qlearning.gif", fps=100)

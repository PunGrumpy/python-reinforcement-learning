import numpy as np
import random
import matplotlib.pyplot as plt


class ReinforcementAgent:
    def __init__(self, num_arms=10, epsilon=0.1):
        self.num_arms = num_arms
        self.arms = np.random.rand(num_arms)  # สุ่มค่า prob ของแต่ละ arm
        self.epsilon = epsilon
        self.action_value = np.array([np.random.randint(0, (num_arms + 1)), 0]).reshape(
            1, 2
        )
        # action value คือ ค่า reward ที่ได้จากการเลือก arm นั้นๆ และ arm ที่เลือก (เริ่มต้นเลือก arm 0 และ reward 0)
        # มีขนาด 1x2 เพราะเก็บค่า arm และ reward ของ arm นั้นๆ

    def reward(self, prob):
        # หาค่า reward โดยการสุ่มตัวเลข 10 ครั้ง ถ้าตัวเลขที่สุ่มออกมาน้อยกว่า prob จะได้ reward 1 แต่ถ้ามากกว่าจะได้ reward 0
        reward = 0
        for i in range(10):
            if random.random() < prob:
                reward += 1
        return reward

    # Epsilon Greedy Algorithm
    def best_arm(self, a):
        # หาว่า arm ไหนให้ reward มากที่สุด (arm คือ ตัวเลขที่สุ่มออกมาจาก np.random.rand(num_arms))
        best_arm = 0
        best_mean = 0
        for u in a:
            avg = np.mean(a[np.where(a[:, 0] == u[0])][:, 1])
            if best_mean < avg:
                best_mean = avg
                best_arm = u[0]
        return best_arm

    # episode คือ จำนวนครั้งที่เลือก arm (เลือก arm 1 ครั้งเท่ากับ 1 episode)
    def train(self, num_episodes=500):
        # ทำการเทรน agent โดยการเลือก arm ที่ดีที่สุด และเก็บค่า reward ที่ได้จากการเลือก arm นั้นๆ
        rewards = []
        for i in range(num_episodes):
            # เอา state ปัจจุบันไปหา action ที่ดีที่สุด
            if random.random() > self.epsilon:
                # ถ้า random.random() มากกว่า epsilon ให้เลือก arm ที่ดีที่สุด (Greedy Exploitation)
                choice = self.best_arm(self.action_value)
                # หาค่า reward ของ arm ที่เลือก
                this_av = np.array([[choice, self.reward(self.arms[choice])]])
                # เก็บค่า arm และ reward ของ arm นั้นๆ
                self.action_value = np.concatenate((self.action_value, this_av), axis=0)
            else:
                # ถ้า random.random() น้อยกว่า epsilon ให้เลือก arm ที่สุ่มมา (Exploration)
                choice = np.where(self.arms == np.random.choice(self.arms))[0][0]
                this_av = np.array([[choice, self.reward(self.arms[choice])]])
                self.action_value = np.concatenate((self.action_value, this_av), axis=0)

            running_mean = np.mean(self.action_value[:, 1])
            rewards.append(running_mean)

        return rewards

    def plot_rewards(self, rewards):
        # ทำการ plot กราฟ reward ที่ได้จากการเทรน agent
        plt.xlabel("Number of episodes")
        plt.ylabel("Average Reward")
        plt.plot(range(len(rewards)), rewards)
        for i in range(len(rewards)):
            plt.scatter(i, rewards[i])
        plt.savefig("reinforcement-beginner.png")


if __name__ == "__main__":
    agent = ReinforcementAgent()
    rewards = agent.train()
    agent.plot_rewards(rewards)

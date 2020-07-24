import cv2
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


NUM_EPISODE = 10  # 50
RENDER = True


class DQN(nn.Module):

    def __init__(self, lr, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(self.device)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent():
    def __init__(self, model, gamma, epsilon, input_dims, batch_size, n_actions,
                 max_mem_size=100, eps_end=0.01, esp_dec=5e-4):

        self.Q_model = model
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = esp_dec

        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if self.epsilon > np.random.random():
            action = np.random.choice(self.action_space)
            return action

        state = torch.tensor([observation]).to(self.Q_model.device)
        actions = self.Q_model.forward(state)
        action = torch.argmax(actions).item()
        return action

    def learn(self):
        if self.batch_size > self.mem_cntr:
            return

        self.Q_model.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_model.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_model.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_model.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_model.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_model.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_model.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_model.loss(q_target, q_eval).to(self.Q_model.device)
        loss.backward()
        self.Q_model.optimizer.step()
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)


def plot_reward_history(rewards):
    x = [i + 1 for i in range(len(rewards))]

    plt.figure()
    plt.title('Reward History')
    plt.ylabel('Reward')
    plt.ylabel('Episode')
    plt.plot(x, rewards, color='tab:blue')
    plt.show()


def main():
    env = gym.make('CartPole-v0')
    _ = env.reset()

    def get_state():
        state = env.render(mode='rgb_array')
        h, w = state.shape[:2]
        state = cv2.resize(state, (int(h / 2), int(w / 2)), interpolation=cv2.INTER_CUBIC)
        return state.transpose((2, 0, 1))

    state = get_state()
    input_dims = state.shape
    n_action = env.action_space.n

    h, w = input_dims[1:]
    model = DQN(0.003, h, w, n_action)
    agent = Agent(model=model, gamma=0.99, epsilon=1.0, batch_size=64, n_actions=n_action,
                  eps_end=0.01, input_dims=input_dims)
    reward_memory = []

    for i_episode in range(NUM_EPISODE):
        rewards = 0
        done = False
        _ = env.reset()
        state = get_state()

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            state_ = get_state()
            rewards += reward

            agent.store_transition(state, action, reward, state_, done)
            agent.learn()
            state = state_

            if RENDER:
                env.render()

        reward_memory.append(rewards)

    env.close()

    # Plot Results.
    plot_reward_history(reward_memory)


if __name__ == '__main__':
    main()

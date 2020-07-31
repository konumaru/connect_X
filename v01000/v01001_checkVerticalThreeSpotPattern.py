# ref: https://www.kaggle.com/alexisbcook/one-step-lookahead
import os
import sys
import json
import base64
import random
import pickle
import inspect
from tqdm import tqdm

import scipy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import kaggle_environments as kaggle_env


FILE_NAME = str(__file__)
VERSION = str(__file__).split('_')[0]
MODEL_FILEPATH = f'cache/{VERSION}_dq_trainer.pkl'
SUBMISSION_FILENAME = f'../submission/{FILE_NAME}'


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


'''Preprocessing.
'''


def preprocess(obs, col_num, row_num):
    # 状態は自分のチェッカーを１、相手のチェッカーを0.5とした7*6次元の配列で表現する
    state = np.array(obs.board)
    state = state.reshape([col_num, row_num])

    if obs.mark == 1:
        return np.where(state == 2, 0.5, state)
    else:
        result = np.where(state == 2, 1, state)
        return np.where(state == 1, 0.5, state)


'''Train Agent.
'''


class CNN(nn.Module):
    def __init__(self, output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(192, 32)
        self.head = nn.Linear(32, output)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.head(x)
        return x


class DeepQNetworkAgent():
    def __init__(self, env, lr=0.01, min_experiences=100, max_experiences=10_000, channel=1):
        self.env = env
        self.model = CNN(output=7)
        self.teacher_model = CNN(output=7)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterrion = nn.MSELoss()
        self.experience = {'s': [], 'a': [], 'r': [], 'n_s': [], 'done': []}
        self.min_experiences = min_experiences
        self.max_experiences = max_experiences
        self.actions = list(range(self.env.configuration.columns))
        self.col_num = self.env.configuration.columns
        self.row_num = self.env.configuration.rows
        self.channel = channel

    def add_experience(self, exp):
        # 行動履歴の更新
        if len(self.experience['s']) >= self.max_experiences:
            # 行動履歴のサイズが大きすぎる時は古いものを削除
            for key in self.experience.keys():
                self.experience[key].pop(0)

        for key, value in exp.items():
            self.experience[key].append(value)

    def preprocess(self, state):
        # 状態は自分のチェッカーを１、相手のチェッカーを0.5とした7*6次元の配列で表現する
        result = np.array(state.board[:])
        result = result.reshape([self.col_num, self.row_num])

        if state.mark == 1:
            return np.where(result == 2, 0.5, result)
        else:
            result = np.where(result == 2, 1, result)
            return np.where(result == 1, 0.5, result)

    def estimate(self, state):
        # 価値の計算
        return self.model(
            torch.from_numpy(state).view(-1, self.channel, self.col_num, self.row_num).float()
        )

    def feature(self, state):
        # 価値の計算
        return self.teacher_model(
            torch.from_numpy(state).view(-1, self.channel, self.col_num, self.row_num).float()
        )

    def policy(self, state, epsilon):
        # 状態からCNNの出力に基づき、次の行動を選択
        if np.random.random() < epsilon:
            # 探索
            return int(np.random.choice(
                [c for c in range(len(self.actions)) if state.board[c] == 0]
            ))
        else:
            # Actionの価値を取得
            prediction = self.estimate(self.preprocess(state))[0].detach().numpy()
            for i in range(len(self.actions)):
                # ゲーム上選択可能なactionに絞る
                if state.board[i] != 0:
                    prediction[i] = -1e7
            return int(np.argmax(prediction))

    def update(self, gamma):
        # 行動履歴が十分に蓄積されている
        if len(self.experience['s']) < self.min_experiences:
            return
        # 行動履歴から学習用のデータのidをサンプリングする
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=32)
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])
        states_next = np.asarray([self.preprocess(self.experience['n_s'][i]) for i in ids])
        # 価値の計算
        estimateds = self.estimate(states).detach().numpy()
        feature = self.feature(states_next).detach().numpy()
        target = estimateds.copy()
        for idx, i in enumerate(ids):
            a = self.experience['a'][i]
            r = self.experience['r'][i]
            d = self.experience['done'][i]

            if d:
                reward = r
            else:
                reward = r + gamma * np.max(feature[idx])
        # TD誤差を小さくするようにCNNを更新
        self.optimizer.zero_grad()
        loss = self.criterrion(
            torch.tensor(estimateds, requires_grad=True),
            torch.tensor(target, requires_grad=True)
        )
        loss.backward()
        self.optimizer.step()

    def update_teacher(self):
        # 繊維先の価値の更新
        self.teacher_model.load_state_dict(self.model.state_dict())


class DeepQNetworkTrainer():
    def __init__(self, env, epsilon=0.9):
        self.env = env
        self.epsilon = epsilon
        self.agent = DeepQNetworkAgent(env)
        self.reward_log = []
        self.num_column = env.configuration['columns']
        self.num_row = env.configuration['rows']

    def check_spot_pattern(self, state, pattern, mode='v'):
        if mode == 'v':
            state = state
        elif mode == 'h':
            state = state.T

        n_window = len(pattern)
        n_window_list = np.array([
            row[i:i + n_window] for row in state for i in range(len(row) - n_window + 1)
        ])

        num_filled = np.all(n_window_list == pattern, axis=1).sum()
        return num_filled

    def custom_reward(self, state, reward, done):
        my_mark = state['mark']
        enemy_mark = state['mark'] % 2 + 1

        board = np.array(state['board']).reshape(self.num_column, self.num_row)

        # Clipping
        if done:
            if reward == 1:  # 勝ち
                return 10000
            elif reward == 0:  # 負け
                return -10000
            else:  # 引き分け
                return 5000
        else:
            score = -0.05
            # Check Own win patterns
            patterns = np.array([
                [my_mark, my_mark, my_mark, 0],
                [my_mark, my_mark, 0, my_mark],
                [my_mark, 0, my_mark, my_mark],
                [0, my_mark, my_mark, my_mark],
            ])
            for pattern in patterns:
                score += self.check_spot_pattern(board, pattern, mode='v')
            # Check Enemy win patterns
            patterns = np.array([
                [enemy_mark, enemy_mark, enemy_mark, 0],
                [enemy_mark, enemy_mark, 0, enemy_mark],
                [enemy_mark, 0, enemy_mark, enemy_mark],
                [0, enemy_mark, enemy_mark, enemy_mark],
            ])
            for pattern in patterns:
                score -= 100 * self.check_spot_pattern(board, pattern, mode='v')

            return score

    def train(self, trainer, epsilon_decay_rate=0.9999,
              min_epsilon=0.01, episode_cnt=100, gamma=0.6):
        cnt = 0
        for episode in tqdm(range(episode_cnt)):
            rewards = []
            state = trainer.reset()  # ゲーム環境リセット
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay_rate)

            while not self.env.done:
                # どの列にドロップするか決める
                action = self.agent.policy(state, self.epsilon)
                prev_state = state
                state, reward, done, _ = trainer.step(action)
                reward = self.custom_reward(state, reward, done)
                # 行動履歴の蓄積
                exp = {'s': prev_state, 'a': action, 'r': reward, 'n_s': state, 'done': done}
                self.agent.add_experience(exp)
                # 価値評価の更新
                self.agent.update(gamma)
                cnt += 1
                if cnt % 100 == 0:
                    # 遷移先価値計算用の更新
                    self.agent.update_teacher()
            self.reward_log.append(reward)


def train_agent():
    env = kaggle_env.make("connectx", debug=False)
    trainer = env.train([None, "negamax"])
    print(json.dumps(env.configuration, indent=2))

    if os.path.exists(MODEL_FILEPATH):
        dq_trainer = load_pickle(MODEL_FILEPATH)
    else:
        dq_trainer = DeepQNetworkTrainer(env)
        dq_trainer.train(trainer, episode_cnt=30000)
        # save cache.
        dump_pickle(dq_trainer, MODEL_FILEPATH)

    # dump reward log.
    reward_log = pd.DataFrame({'Average Reward': dq_trainer.reward_log})
    plt.figure()
    reward_log.rolling(300).mean().plot(
        figsize=(10, 5),
        title='Average Reward by rolling t300.'
    )
    plt.savefig(f'train_log/{VERSION}_reward_lr_history.png')


'''Create Submission File.
- 必要なライブラリ、クラス、関数をsubmissionファイルに書き出す
'''

libs_source = '''\
import io
import base64
import pickle
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
'''

write_functions = [CNN, preprocess]

agent_source = '''\
def load_model():
    model = CNN(7)
    encoded_weights = "{model_state_dict_bin}".encode()
    weights = pickle.loads(base64.b64decode(encoded_weights))
    model.load_state_dict(weights)
    return model


model = load_model()


def agent(observation, config):
    col_num = config.columns
    row_num = config.rows
    channel = 1

    state = preprocess(observation, col_num, row_num)
    prediction = model(
        torch.from_numpy(state).view(-1, channel, col_num, row_num).float()
    ).detach().numpy()
    action = int(np.argmax(prediction))

    # 選んだActionが、ゲーム上選べない場合
    if observation.board[action] != 0:
        return random.choice([c for c in range(config.columns) if observation.board[c] == 0])

    return action
'''


def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function) + '\n\n')


def create_submission_file():
    dq = load_pickle(MODEL_FILEPATH)
    state_dict = dq.agent.model.state_dict()
    model_state_dict_bin = base64.b64encode(pickle.dumps(state_dict)).decode("utf-8")

    with open(SUBMISSION_FILENAME, "w") as f:
        f.write(libs_source + '\n\n')

    for wf in write_functions:
        write_agent_to_file(wf, SUBMISSION_FILENAME)

    formated_agent_source = agent_source.format(model_state_dict_bin=model_state_dict_bin)
    with open(SUBMISSION_FILENAME, "a") as f:
        f.write(formated_agent_source)

    # check can submission.
    env = kaggle_env.make("connectx", debug=True)
    env.run([SUBMISSION_FILENAME, SUBMISSION_FILENAME])
    print("Success" if env.state[0].status == env.state[1].status == "DONE" else "Failed")


'''Evaluation Agent. (vs Ramdon Agent)
'''


def evaluation():
    result = np.array(
        kaggle_env.evaluate("connectx", [SUBMISSION_FILENAME, "random"], num_episodes=300)
    )
    win_rate = np.mean(result[:, 0] == 1)
    print(f'Average Win Ratio: {win_rate}')

    n_win = sum(result[:, 0] == 1)
    n_lose = sum(result[:, 0] == -1)

    x = np.linspace(0, 1, 1002)[1:-1]
    dist = scipy.stats.beta(n_win + 1, n_lose + 1)

    plt.figure()
    plt.title('Beta Distribution of Win Rate.')
    plt.plot(x, dist.pdf(x))
    plt.savefig(f'evaluation/{VERSION}_beta_distribution_of_win_rate.png')


def main():
    print('\n\n--- Train Agent ---\n\n')
    train_agent()

    print('\n\n--- Create Submission File ---\n\n')
    create_submission_file()

    print('\n\n--- Evaluation Agent. (vs Ramdon Agent) ---\n\n')
    evaluation()


if __name__ == '__main__':
    main()

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

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import kaggle_environments as kaggle_env

IS_TEST = False

FILE_NAME = str(__file__)
VERSION = str(__file__).split('_')[0]
MODEL_FILEPATH = f'cache/{VERSION}_dq_trainer.pkl'
SUBMISSION_FILENAME = f'../submission/{FILE_NAME}'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'cuda available: {DEVICE}')


'''Preprocessing.
'''


def preprocess(state: dict) -> np.array:
    row = 6
    column = 7

    own_mark = state['mark']
    baord = state['board']

    assert len(baord) == 42, "Board length is 7 * 6 or 42."

    board = np.array(baord).reshape([column, row])
    # Convert Own place to 1.0 and Enemy place to 0.5.
    if own_mark == 1:
        return np.where(board == 2, 0.5, board)
    elif own_mark == 2:
        board = np.where(board == 1, 0.5, board)
        return np.where(board == 2, 1, board)


'''Train Agent.
'''


class CNN(nn.Module):
    def __init__(self, num_channel: int, num_action: int):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(192, 32)
        self.fc2 = nn.Linear(32, num_action)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class NeurallNet():
    def __init__(self, num_state: int, hidden_units: list, num_action: int):
        super(NeurallNet).__init__()
        hidden_units = [num_state] + hidden_units
        self.hidden_layers = []
        for i, u in enumerate(hidden_units[:-1]):
            self.hidden_layers.append(nn.Linear(u, hidden_units[i + 1]))
        self.output_layer = nn.Linear(hidden_units[-1], num_action)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class DDQNAgent():
    def __init__(self, env, num_action, lr=1e-3,
                 batch_size=100, max_mem_size=10_000, num_channel=1):
        self.env = env
        self.num_action = num_action
        self.batch_size = batch_size
        self.num_channel = num_channel

        self.num_row = self.env.configuration.rows
        self.num_column = self.env.configuration.columns

        self.model = CNN(num_channel, num_action).to(DEVICE)
        self.target_model = CNN(num_channel, num_action).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.mem_cntr = 0
        self.mem_size = max_mem_size
        self.memory = {
            's': np.zeros((self.mem_size, *(self.num_column, self.num_row)), dtype=np.float32),
            'a': np.zeros(self.mem_size, dtype=np.float32),
            'r': np.zeros(self.mem_size, dtype=np.float32),
            'n_s': np.zeros((self.mem_size, *(self.num_column, self.num_row)), dtype=np.float32),
            'done': np.zeros(self.mem_size, dtype=np.float32)
        }

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_cntr % self.mem_size
        self.memory['s'][idx] = state
        self.memory['a'][idx] = action
        self.memory['r'][idx] = reward
        self.memory['n_s'][idx] = next_state
        self.memory['done'][idx] = done
        self.mem_cntr += 1

    def q_values(self, state: np.array):
        self.model.eval()
        return self.model(
            torch.from_numpy(state).view(
                -1, self.num_channel, self.num_column, self.num_row
            ).float().to(DEVICE)
        )

    def target_q_values(self, state: np.array):
        self.target_model.eval()
        return self.target_model(
            torch.from_numpy(state).view(
                -1, self.num_channel, self.num_column, self.num_row
            ).float().to(DEVICE)
        )

    def act(self, state: np.array, epsilon: float):
        if np.random.random() < epsilon:
            return int(np.random.choice(
                [c for c in range(self.num_action) if np.min(state, axis=1)[c] == 0]
            ))
        else:
            prediction = self.q_values(state)[0].detach().numpy()
            for i in range(self.num_action):
                # ゲーム上選択可能なactionに絞る
                if np.min(state, axis=1)[i] != 0:
                    prediction[i] = -1e7
            return int(np.argmax(prediction))

    def update(self, gamma: float):
        # メモリーがバッチサイズ以下であれば学習しない
        if self.mem_cntr < self.batch_size:
            return
        # ReplayMemory, メモリーからランダムサンプリングを行う
        max_mem = min(self.mem_cntr, self.mem_size)
        batch_idx = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.memory['s'][batch_idx]
        actions = self.memory['a'][batch_idx]
        rewards = self.memory['r'][batch_idx]
        n_states = self.memory['n_s'][batch_idx]
        done = self.memory['done'][batch_idx]

        q_eval = self.q_values(states).detach().numpy()
        q_eval = np.array([q[a] for q, a in zip(q_eval, actions.astype(int))])
        q_target = np.max(self.target_q_values(n_states).detach().numpy(), axis=1)
        q_target = np.where(done, rewards, rewards + gamma * q_target)

        self.model.train()
        loss = self.criterion(
            torch.tensor(q_eval, requires_grad=True).to(DEVICE),
            torch.tensor(q_target, requires_grad=True).to(DEVICE).to(DEVICE)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


class AgentTrainer():
    def __init__(self, env, num_action):
        self.agent = DDQNAgent(
            env, num_action,
            lr=1e-3, batch_size=512,
            max_mem_size=10_000, num_channel=1
        )
        self.num_row = env.configuration['rows']
        self.num_column = env.configuration['columns']

        self.train_log = {'epsilon': [], 'reward': []}

    def count_v_and_h_three_spot(self, state: np.array, next_state: np.array, mark: float):
        prev_counter = 0
        next_counter = 0
        # count vertical patterns.
        v_patterns = np.array([
            [0, mark, mark, mark],
            [mark, 0, mark, mark],
            [mark, mark, 0, mark],
            [mark, mark, mark, 0],
        ])

        for v_pattern in v_patterns:
            n_window = len(v_pattern)
            # Prev State Count.
            n_window_list = np.array([
                row[i:i + n_window] for row in state for i in range(len(row) - n_window + 1)
            ])
            prev_counter += np.all(n_window_list == v_pattern, axis=1).sum()
            # Next State Count.
            n_window_list = np.array([
                row[i:i + n_window] for row in next_state for i in range(len(row) - n_window + 1)
            ])
            next_counter += np.all(n_window_list == v_pattern, axis=1).sum()

        # count horizontal pattern.
        h_patterns = np.array([0, mark, mark, mark])
        n_window = len(h_patterns)
        # Prev State Count.
        n_window_list = np.array(
            [row[i:i + n_window] for row in state.T for i in range(len(row) - n_window + 1)]
        )
        prev_counter += np.all(n_window_list == h_patterns, axis=1).sum()
        # Next State Count.
        n_window_list = np.array(
            [row[i:i + n_window] for row in next_state.T for i in range(len(row) - n_window + 1)]
        )
        next_counter += np.all(n_window_list == h_patterns, axis=1).sum()
        return (next_counter - prev_counter)

    def count_block_opponent_win(self, state: np.array, next_state: np.array,
                                 my_mark: float, opp_mark: float):
        prev_counter = 0
        next_counter = 0
        # count vertical patterns.
        v_patterns = np.array([
            [my_mark, opp_mark, opp_mark, opp_mark],
            [opp_mark, my_mark, opp_mark, opp_mark],
            [opp_mark, opp_mark, my_mark, opp_mark],
            [opp_mark, opp_mark, opp_mark, my_mark],
        ])

        for v_pattern in v_patterns:
            n_window = len(v_pattern)
            # Prev State Count.
            n_window_list = np.array([
                row[i:i + n_window] for row in state for i in range(len(row) - n_window + 1)
            ])
            prev_counter += np.all(n_window_list == v_pattern, axis=1).sum()
            # Next State Count.
            n_window_list = np.array([
                row[i:i + n_window] for row in next_state for i in range(len(row) - n_window + 1)
            ])
            next_counter += np.all(n_window_list == v_pattern, axis=1).sum()

        # count horizontal pattern.
        h_patterns = np.array([my_mark, opp_mark, opp_mark, opp_mark])
        n_window = len(h_patterns)
        # Prev State Count.
        n_window_list = np.array(
            [row[i:i + n_window] for row in state.T for i in range(len(row) - n_window + 1)]
        )
        prev_counter += np.all(n_window_list == h_patterns, axis=1).sum()
        # Next State Count.
        n_window_list = np.array(
            [row[i:i + n_window] for row in next_state.T for i in range(len(row) - n_window + 1)]
        )
        next_counter += np.all(n_window_list == h_patterns, axis=1).sum()
        return (next_counter - prev_counter)

    def custom_reward(self, state: np.array, next_state: np.array, reward: int, done: bool):
        """
        # 報酬設計
        - 自駒が縦・横３つ揃う -> +100
        - 他駒が縦・横３つ揃う -> -100
        - 他駒が４つ揃いそうなとき防ぐ -> +100
        - 他駒が斜めに３つ揃う
        - 自駒が斜めに３つ揃う
        """
        my_mark = 1.0
        opp_mark = 0.5
        # Clipping
        if done:
            if reward == 1:  # 勝ち
                score = 10
            elif reward == 0:  # 負け
                score = -10
            else:  # 引き分け
                score = 0
        else:
            score = -0.05
            # Long-term match cost
            # score -= 0.05 * np.sum(state != 0)
            # Check three spot patterns
            score += 1 * self.count_v_and_h_three_spot(state, next_state, my_mark)
            score -= 3 * self.count_v_and_h_three_spot(state, next_state, opp_mark)
            # Check Blocked the opponent's win
            score += 5 * self.count_block_opponent_win(state, next_state, my_mark, opp_mark)
        return score

    def train(self, env, max_epsilon=0.9, epsilon_decay_rate=0.9999, min_epsilon=1e-2,
              num_episode=1000, gamma=0.9, verbose=50):
        epsilon = max_epsilon
        episode_cntr = 0
        num_digit = len(str(num_episode))

        for i_eps in range(num_episode):
            if (i_eps + 1) % verbose == 0 and i_eps != 0:
                latest_avg_reward = np.mean(self.train_log['reward'][:-100])
                print(f"{i_eps+1: >{num_digit}} / {num_episode:}:  ", end='')
                print(f"reward  {latest_avg_reward:.4f}  ", end='')
                print(f"epsilon {epsilon:.4f}.")

            # match_type = [
            #     [None, "negamax"], [None, "random"],
            #     ["negamax", None], ["random", None]
            # ]
            # choosen_type = np.random.choice(len(match_type), p=[0.25, 0.25, 0.25, 0.25])
            match_type = [
                [None, "random"], ["random", None]
            ]
            choosen_type = np.random.choice(len(match_type), p=[0.5, 0.5])
            trainer = env.train(match_type[choosen_type])

            epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
            state = trainer.reset()
            state = preprocess(state)

            while True:
                action = self.agent.act(state, epsilon)
                next_state, reward, done, _ = trainer.step(action)
                next_state = preprocess(next_state)
                reward = self.custom_reward(state, next_state, reward, done)

                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.update(gamma)

                self.train_log['epsilon'].append(epsilon)
                self.train_log['reward'].append(reward)

                state = next_state

                if episode_cntr % 100 == 0:
                    self.agent.update_target()

                if done:
                    break


def save_train_log(train_log, t_rool=5000):
    df = pd.DataFrame(train_log)
    # df.to_csv(f'train_log/{VERSION}_train_log.csv', index=False)
    df = df.rolling(t_rool).mean()

    fig, ax1 = plt.subplots()
    plt.title(f'Reward and Epsilon Transition (rolling t{t_rool}.).')
    plt.xlabel('Episode')
    ax2 = ax1.twinx()

    ax1.set_ylabel('Reward')
    ax1.plot(df.index, df["reward"], color='tab:blue', label="Rewards")
    ax2.set_ylabel('Epsilon')
    ax2.plot(df.index, df["epsilon"], color='tab:green', label="Epsilon")

    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()

    ax1.legend(handler1 + handler2, label1 + label2, loc=3)
    plt.savefig(f'train_log/{VERSION}_reward_lr_history.png')


def train_agent():
    env = kaggle_env.make("connectx", debug=False)
    num_action = env.configuration.columns
    print(json.dumps(env.configuration, indent=2), '\n')

    if os.path.exists(MODEL_FILEPATH) and not IS_TEST:
        dq_trainer = torch.load(MODEL_FILEPATH)
    else:
        verbose = 10 if IS_TEST else 1000
        train_esp_cnt = 100 if IS_TEST else 100_000

        dq_trainer = AgentTrainer(env, num_action)
        dq_trainer.train(
            env, max_epsilon=0.9, epsilon_decay_rate=0.9999, min_epsilon=1e-2,
            num_episode=train_esp_cnt, gamma=0.9, verbose=verbose
        )
        # save cache.
        torch.save(dq_trainer, MODEL_FILEPATH)

    save_train_log(dq_trainer.train_log)


'''Create Submission File.
- 必要なライブラリ、クラス、関数をsubmissionファイルに書き出す
'''


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

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
'''

write_functions = [CNN, preprocess]

agent_source = '''\
def load_model():
    model = CNN(1, 7).to(DEVICE)
    encoded_weights = "{model_state_dict_bin}".encode()
    weights = pickle.loads(base64.b64decode(encoded_weights))
    model.load_state_dict(weights)
    return model


model = load_model()


def agent(observation, config):
    col_num = config.columns
    row_num = config.rows
    channel = 1

    state = preprocess(observation)
    prediction = model(
        torch.from_numpy(state).view(-1, channel, col_num, row_num).float().to(DEVICE)
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


def submission():
    dq_trainer = torch.load(MODEL_FILEPATH)
    state_dict = dq_trainer.agent.model.state_dict()
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


def calc_beta(result):
    n_win = sum(result[:, 0] == 1)
    n_lose = sum(result[:, 0] == -1)

    x = np.linspace(0, 1, 1002)[1:-1]
    dist = scipy.stats.beta(n_win + 1, n_lose + 1)

    return (x, dist.pdf(x))


def evaluation():
    num_esp = 10 if IS_TEST else 100

    random_result = np.array(
        kaggle_env.evaluate("connectx", [SUBMISSION_FILENAME, "random"], num_episodes=num_esp)
    )
    random_win_rate = np.mean(random_result[:, 0] == 1)
    print(f'Average Win Ratio VS Random: {random_win_rate}')

    negamax_result = np.array(
        kaggle_env.evaluate("connectx", [SUBMISSION_FILENAME, "negamax"], num_episodes=num_esp)
    )
    negamax_win_rate = np.mean(negamax_result[:, 0] == 1)
    print(f'Average Win Ratio VS Negamax: {negamax_win_rate}')

    plt.figure()
    plt.title('Beta Distribution of Win Rate.')
    plt.plot(*calc_beta(random_result), color='tab:blue', label=f'random, {random_win_rate}')
    plt.plot(*calc_beta(negamax_result), color='tab:orange', label=f'negamax, {negamax_win_rate}')
    plt.legend()
    plt.savefig(f'evaluation/{VERSION}_beta_distribution_of_win_rate.png')


def main():
    print('\n\n--- Train Agent ---\n\n')
    train_agent()

    print('\n\n--- Create Submission File ---\n\n')
    submission()

    print('\n\n--- Evaluation Agent. (vs Ramdon Agent) ---\n\n')
    evaluation()


if __name__ == '__main__':
    main()

import os
import sys
import json
import inspect
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


from kaggle_environments import make, utils
from kaggle_environments import evaluate

IS_TEST = False

LOG_DIR = 'log/v02000/'
SUBMISSION_FILENAME = "Agent/v02000_ql_agent.py"


class QTable():
    def __init__(self, actions):
        self.Q = {}
        self.actions = actions

    def get_state_key(self, state):
        # 16進数で状態のkeyを作る
        board = state.board[:]
        board.append(state.mark)
        state_key = np.array(board).astype(str)
        return hex(int(''.join(state_key), 3))[2:]

    def get_q_values(self, state):
        # 状態に対して、全actionのQ値の配列を出力
        state_key = self.get_state_key(state)
        if state_key not in self.Q.keys():
            # 過去に観測されたことのないstateの場合
            self.Q[state_key] = [0] * len(self.actions)
        return self.Q[state_key]

    def update(self, state, action, add_q):
        # Q値を更新
        state_key = self.get_state_key(state)
        self.Q[state_key] = [
            q + add_q if idx == action else q for idx, q in enumerate(self.Q[state_key])
        ]


class QLearningAgent():
    def __init__(self, env, epsilon=0.99):
        self.env = env
        self.actions = list(range(self.env.configuration.columns))
        self.q_table = QTable(self.actions)
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, state):
        if np.random.random() < self.epsilon:
            # 一定の確率で、ランダムにactionを選択する
            return random.choice([c for c in range(len(self.actions)) if state.board[c] == 0])
        else:
            # 選択可能な範囲で、Q値が最大なactionを選択する
            q_values = self.q_table.get_q_values(state)
            selected_items = [
                q if state.board[idx] == 0 else -1e7 for idx, q in enumerate(q_values)
            ]
            return int(np.argmax(selected_items))

    def custom_reward(self, reward, done):
        if done:
            if reward == 1:
                # 勝利した場合
                return 20
            elif reward == 0:
                # 敗北した場合
                return -20
            else:
                # 引き分けだった場合
                return 10
        else:
            # 勝負がついていない場合
            return -0.05

    def learn(self, trainer, episode_cnt=10000, gamma=0.6, lr=0.3,
              epsilon_decay_rate=0.999, min_epsilon=0.1):
        for episode in tqdm(range(episode_cnt)):
            # ゲーム環境のリセット
            state = trainer.reset()
            # epsilonを徐々に小さくする
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay_rate)

            while not self.env.done:
                # どの列にdropするのか決める
                action = self.policy(state)
                next_state, reward, done, info = trainer.step(action)
                reward = self.custom_reward(reward, done)
                # 誤差を計算してQTableを更新する
                gain = reward + gamma * max(self.q_table.get_q_values(next_state))
                estimate = self.q_table.get_q_values(state)[action]
                add_q = lr * (gain - estimate)
                self.q_table.update(state, action, add_q)

            self.reward_log.append(reward)


def check_clear_submission(filename):
    out = sys.stdout
    submission = utils.read_file(filename)
    agent = utils.get_last_callable(submission)
    sys.stdout = out

    env = make("connectx", debug=True)
    env.run([agent, agent])
    print("Success" if env.state[0].status == env.state[1].status == "DONE" else "Failed")


def main():
    env = make("connectx", debug=False)
    print(json.dumps(env.configuration, indent=2))

    trainer = env.train([None, "random"])
    agent = QLearningAgent(env)
    agent.learn(trainer)

    sns.set(style='darkgrid')
    pd.DataFrame({'AverageReward': agent.reward_log}).rolling(500).mean().plot(figsize=(10, 5))
    plt.savefig(LOG_DIR + 'reward_log.png')

    tmp_dict_q_table = agent.q_table.Q.copy()
    dict_q_table = dict()

    # 学習したQテーブルで、一番Q値の大きいActionに置き換える
    for k in tmp_dict_q_table.keys():
        if np.count_nonzero(tmp_dict_q_table[k]) > 0:
            dict_q_table[k] = np.argmax(tmp_dict_q_table[k]).astype('int8')

    my_agent = '''
def my_agent(observation, configuration):
    from random import choice
    # 作成したテーブルを文字列に変換して、Pythonファイル上でdictとして扱えるようにする
    q_table = ''' \
    + str(dict_q_table).replace(' ', '') \
        + '''
    board = observation.board[:]
    board.append(observation.mark)
    state_key = list(map(str, board))
    state_key = hex(int(''.join(state_key), 3))[2:]
    # Qテーブルに存在しない状態の場合
    if state_key not in q_table.keys():
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
    # Qテーブルから最大のQ値をとるActionを選択
    action = q_table[state_key]
    # 選んだActionが、ゲーム上選べない場合
    if observation.board[action] != 0:
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
    return action
'''

    with open(SUBMISSION_FILENAME, 'w') as f:
        f.write(my_agent)

    check_clear_submission(SUBMISSION_FILENAME)

    print(evaluate("connectx", [SUBMISSION_FILENAME, "random"], num_episodes=3))


if __name__ == '__main__':
    main()

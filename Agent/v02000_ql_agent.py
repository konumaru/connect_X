
def my_agent(observation, configuration):
    from random import choice
    # 作成したテーブルを文字列に変換して、Pythonファイル上でdictとして扱えるようにする
    q_table = {'1':5}
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

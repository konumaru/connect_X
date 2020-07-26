import os
import sys
import json
import inspect

from kaggle_environments import make, utils
from kaggle_environments import evaluate

IS_TEST = False
SUB_FILENAME = "Agent/v01000_simple_agent.py"


def my_agent(state, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if state.board[c] == 0])


def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))


def check_clear_submission(filename):
    out = sys.stdout
    submission = utils.read_file(filename)
    agent = utils.get_last_callable(submission)
    sys.stdout = out

    env = make("connectx", debug=True)
    env.run([agent, agent])
    print("Success" if env.state[0].status == env.state[1].status == "DONE" else "Failed")


def main():
    env = make('connectx', debug=True)
    print(env.render(mode='ansi'))
    print(json.dumps(env.configuration, indent=2))

    trainer = env.train([None, 'random'])
    state = trainer.reset()
    print(f'board: {state.board}')
    print(f'mark: {state.mark}')

    while not env.done:
        state, reward, done, info = trainer.step(0)
        print(f"reward: {reward}, done: {done}, info: {info}")
        board = state.board

    print(evaluate("connectx", [my_agent, "random"], num_episodes=3))

    write_agent_to_file(my_agent, SUB_FILENAME)

    if not IS_TEST:
        check_clear_submission(SUB_FILENAME)


if __name__ == '__main__':
    main()

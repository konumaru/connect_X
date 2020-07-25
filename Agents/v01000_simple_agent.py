def my_agent(state, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if state.board[c] == 0])
def my_agent(state, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if state.board[c] == 0])

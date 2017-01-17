"""
tic tac toe

0: vacant
1: first player
2: second player
"""
import numpy as np

def board_to_int(board):
    s = 0
    for i in range(9):
        s += board[i] * (3 ** i)
    return s

def board_to_possible_hands(board):
    return [i for i in range(9) if board[i] == 0]

def init_board():
    return np.zeros(9, dtype=np.int)

def init_Q():
    return [0] * (3 ** 9 * 9)

def policy_random(board):
    from random import choice
    actions = board_to_possible_hands(board)
    return choice(actions)

LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]
def is_win(board):
    for line in LINES:
        a, b, c = board[line]
        if a != 0 and a == b and a == c:
            return a
    return 0

def print_board(board):
    s = ['.ox'[x] for x in board]
    print ' '.join(s[0:3])
    print ' '.join(s[3:6])
    print ' '.join(s[6:9])
    print


class Environment(object):
    def __init__(self, policy=policy_random):
        self.board = init_board()
        self.op_policy = policy
        self.result_log =[]

    def __call__(self, action):
        if self.board[action] != 0:
            # illegal move
            raise NotImplementerError
            return

        self.board[action] = 1
        b = is_win(self.board)
        if b:
            self.board = init_board()
            self.result_log.append(1)
            return (self.board, +1)

        if not board_to_possible_hands(self.board):
            self.board = init_board()
            self.result_log.append(0)
            return (self.board, -1)

        op_action = self.op_policy(self.board)
        self.board[op_action] = 2
        b = is_win(self.board)
        if b:
            self.result_log.append(2)
            self.board = init_board()
            return (self.board, -1)

        return (self.board, 0)


def play(policy1, policy2=policy_random, to_print=False):
    board = init_board()
    result = 0
    for i in range(9):
        if i % 2 == 0:
            a = policy1(board)
            board[a] = 1
        else:
            a = policy2(board)
            board[a] = 2

        if to_print:
            print_board(board)

        b = is_win(board)
        if b:
            result = b
            break
    return result


from collections import Counter
if not"ex1":
    print Counter(
        play(policy_random) for i in range(10000))

class Greedy(object):
    def __init__(self):
        self.Qtable = init_Q()

    def __call__(self, board):
        from random import choice
        s = board_to_int(board)
        actions = board_to_possible_hands(board)
        qa = [(self.Qtable[s * 9 + a], a) for a in actions]
        bestQ, bestA = max(qa)
        bextQ, bestA = choice([(q, a) for (q, a) in qa if q == bestQ])
        return bestA

def board_to_state(board):
    return board_to_int(board)

alpha = 0.5
gamma = 0.9
batch_width = 100
num_batch = 100
num_result = batch_width * num_batch
environment = Environment()
policy = Greedy()
action = policy(environment.board)
state = board_to_state(environment.board)
while True:
    next_board, reward = environment(action)
    #print_board(next_board)
    next_state = board_to_state(next_board)

    # determine a'
    next_action = policy(next_board)
    nextQ = policy.Qtable[next_state * 9 + next_action]

    # update Q(s, a)
    s_a = state * 9 + action
    Qsa = policy.Qtable[s_a]
    estimated_reward = reward + gamma * nextQ
    diff = estimated_reward - Qsa
    policy.Qtable[s_a] += alpha * diff

    state = next_state
    action = next_action
    if len(environment.result_log) == num_result:
        break

vs = []
for i in range(num_batch):
    c = Counter(environment.result_log[batch_width * i : batch_width * (i + 1)])
    print c
    vs.append(float(c[1]) / batch_width)


import matplotlib.pyplot as plt
plt.plot([0.58] * len(vs), label = "baseline")
plt.plot(vs, label = "Sarsa")
plt.xlabel("iteration")
plt.ylabel("Prob. of win")
plt.legend(loc = 4)
plt.savefig('sarsa.png')

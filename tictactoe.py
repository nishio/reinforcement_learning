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
            self.board = init_board()
            self.result_log.append(2)
            return (self.board, -1)

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
        #actions = board_to_possible_hands(board)
        actions = range(9)
        qa = [(self.Qtable[s * 9 + a], a) for a in actions]
        bestQ, bestA = max(qa)
        bextQ, bestA = choice([(q, a) for (q, a) in qa if q == bestQ])
        return bestA


def board_to_state(board):
    return board_to_int(board)


def sarsa(alpha, policyClass=Greedy):
    global environment, policy
    gamma = 0.9
    num_result = batch_width * num_batch
    environment = Environment()
    policy = policyClass()
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
    return vs

def get_Qkey(state, action):
    return state * 9 + action
    #return (state, action_to_int(action))

def qlearn(alpha, policyClass=Greedy):
    global environment, policy
    gamma = 0.9
    num_result = batch_width * num_batch
    environment = Environment()
    policy = policyClass()

    state = board_to_state(environment.board)
    while True:
        action = policy(environment.board)
        next_board, reward = environment(action)
        next_state = board_to_state(next_board)

        # update Q(s, a)
        maxQ = max(policy.Qtable[get_Qkey(next_state, a)]
                   for a in board_to_possible_hands(next_board))
        s_a = get_Qkey(state, action)

        Qsa = policy.Qtable[s_a]
        estimated_reward = reward + gamma * maxQ
        diff = estimated_reward - Qsa
        policy.Qtable[s_a] += alpha * diff

        state = next_state

        if len(environment.result_log) == num_result:
            break

    vs = []
    for i in range(num_batch):
        c = Counter(environment.result_log[batch_width * i : batch_width * (i + 1)])
        print c
        vs.append(float(c[1]) / batch_width)
    return vs


batch_width = 100
num_batch = 100

import matplotlib.pyplot as plt
plt.clf()

if 0:
    vs1 = sarsa(0.5)
    vs2 = sarsa(0.05)
    vs3 = sarsa(0.005)

    plt.plot([0.58] * len(vs1), label = "baseline")
    plt.plot(vs1, label = "Sarsa(0.5)")
    plt.plot(vs2, label = "Sarsa(0.05)")
    plt.plot(vs3, label = "Sarsa(0.005)")
    plt.xlabel("iteration")
    plt.ylabel("Prob. of win")
    plt.legend(loc = 4)
    plt.savefig('sarsa.png')


if 1:
    vs1 = qlearn(0.5)
    vs2 = qlearn(0.05)
    vs3 = qlearn(0.005)
    plt.plot([0.58] * len(vs1), label = "baseline")
    plt.plot(vs1, label = "Qlearn(0.5)")
    plt.plot(vs2, label = "Qlearn(0.05)")
    plt.plot(vs3, label = "Qlearn(0.005)")
    plt.xlabel("iteration")
    plt.ylabel("Prob. of win")
    plt.legend(loc = 4)
    plt.savefig('qlearn.png')

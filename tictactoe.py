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

class QRaw(object):
    def __init__(self):
        self.value = [0] * (3 ** 9 * 9)
    def get(self, s, a):
        return self.value[s * 9 + a]
    def set(self, s, a, v):
        self.value[s * 9 + a] = v
    def board_to_state(self, board):
        return board_to_int(board)


class QSym(QRaw):
    "Compress states using symmetry"
    def board_to_state(self, board):
        buf = []
        for i in range(4):
            buf.append(board_to_int(board))
            # rotate 90 ccw
            board = board[[2, 5, 8, 1, 4, 7, 0, 3, 6]]
        # mirror
        board = board[[2, 1, 0, 5, 4, 3, 8, 7, 6]]
        for i in range(4):
            buf.append(board_to_int(board))
            # rotate 90 ccw
            board = board[[2, 5, 8, 1, 4, 7, 0, 3, 6]]

        return min(buf)


class QLine(QRaw):
    "Compress states using symmetry"
    def board_to_state(self, board):
        raise NotImplementedError
        return min(buf)



def policy_random(env):
    from random import choice
    actions = board_to_possible_hands(env.board)
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
            return (self, -1)

        self.board[action] = 1
        b = is_win(self.board)
        if b:
            self.board = init_board()
            self.result_log.append(1)
            return (self, +1)

        if not board_to_possible_hands(self.board):
            self.board = init_board()
            self.result_log.append(0)
            return (self, -1)

        op_action = self.op_policy(self)
        self.board[op_action] = 2
        b = is_win(self.board)
        if b:
            self.result_log.append(2)
            self.board = init_board()
            return (self, -1)

        return (self, 0)


def play(policy1, policy2=policy_random, to_print=False):
    env = Environment()
    result = 0
    for i in range(9):
        if i % 2 == 0:
            a = policy1(env)
            env.board[a] = 1
        else:
            a = policy2(env)
            env.board[a] = 2

        if to_print:
            print_board(env.board)

        b = is_win(env.board)
        if b:
            result = b
            break
    return result


from collections import Counter
if not"ex1":
    print Counter(
        play(policy_random) for i in range(10000))

class Greedy(object):
    def __init__(self, QClass=QRaw):
        self.Qtable = QRaw()

    def __call__(self, env):
        from random import choice
        s = board_to_int(env.board)
        actions = get_available_actions(env)
        qa = [(self.Qtable.get(s, a), a) for a in actions]
        bestQ, bestA = max(qa)
        bextQ, bestA = choice([(q, a) for (q, a) in qa if q == bestQ])
        return bestA

def get_available_actions(env):
    return range(9)
    # return board_to_possible_hands(board)

class EpsilonGreedy(object):
    def __init__(self, eps=0.1):
        self.Qtable = init_Q()
        self.eps = eps

    def __call__(self, env):
        from random import choice, random
        s = board_to_int(env.board)
        if random() < self.eps:
            actions = get_available_actions(env)
            return choice(actions)

        actions = get_available_actions(env)
        qa = [(self.Qtable[get_Qkey(s, a)], a) for a in actions]
        bestQ, bestA = max(qa)
        bextQ, bestA = choice([(q, a) for (q, a) in qa if q == bestQ])
        return bestA



def sarsa(alpha, policyClass=Greedy):
    global environment, policy
    gamma = 0.9
    num_result = batch_width * num_batch
    environment = Environment()
    policy = policyClass()
    action = policy(environment)
    state = policy.Qtable.board_to_state(environment.board)
    prev_num_battle = 0
    win_ratios = []
    while True:
        next_env, reward = environment(action)
        next_state = policy.Qtable.board_to_state(next_env.board)

        # determine a'
        next_action = policy(next_env)
        nextQ = policy.Qtable.get(next_state, next_action)

        # update Q(s, a)
        s_a = state * 9 + action
        Qsa = policy.Qtable.get(state, action)
        estimated_reward = reward + gamma * nextQ
        diff = estimated_reward - Qsa
        policy.Qtable.set(state, action, Qsa + alpha * diff)

        state = next_state
        action = next_action

        num_battle = len(environment.result_log)
        if num_battle > prev_num_battle:
            if num_battle % 10 == 0:
                c = Counter(
                    play(policy, policy_random) for i in range(100))
                win_ratios.append(float(c[1]) / 100)
        if num_battle == num_result:
            break
        prev_num_battle = num_battle

    #vs = []
    #for i in range(num_batch):
    #    c = Counter(environment.result_log[batch_width * i : batch_width * (i + 1)])
    #    print c
    #    vs.append(float(c[1]) / batch_width)
    #return vs

    return win_ratios

def calc_win_ratios_from_result_log(env):
    vs = []
    for i in range(num_batch):
        c = Counter(environment.result_log[
            batch_width * i :
            batch_width * (i + 1)])
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

    state = policy.Qtable.board_to_state(environment.board)
    prev_num_battle = 0
    win_ratios = []
    while True:
        action = policy(environment)
        next_env, reward = environment(action)
        next_state = policy.Qtable.board_to_state(next_env.board)

        # update Q(s, a)
        maxQ = max(policy.Qtable.get(next_state, a)
                   for a in get_available_actions(next_env))

        Qsa = policy.Qtable.get(state, action)
        estimated_reward = reward + gamma * maxQ
        diff = estimated_reward - Qsa
        policy.Qtable.set(state, action, Qsa + alpha * diff)

        state = next_state

        num_battle = len(environment.result_log)
        if num_battle > prev_num_battle:
            if num_battle % 10 == 0:
                c = Counter(
                    play(policy, policy_random) for i in range(100))
                win_ratios.append(float(c[1]) / 100)
        if num_battle == num_result:
            break
        prev_num_battle = num_battle

    vs = []
    for i in range(num_batch):
        c = Counter(environment.result_log[batch_width * i : batch_width * (i + 1)])
        print c
        vs.append(float(c[1]) / batch_width)
    return vs


batch_width = 100
num_batch = 100

import matplotlib.pyplot as plt

def plot(seq, name, baseline=None):
    """seq: [(values, label)]"""
    plt.clf()
    if baseline != None:
        plt.plot([0.58] * len(seq[0][0]), label = "baseline")
    for (vs, label) in seq:
        plt.plot(vs, label=label)
    plt.xlabel("iteration")
    plt.ylabel("Prob. of win")
    plt.legend(loc = 4)
    plt.savefig(name)


batch_width = 10
num_batch = 100
result = [
    (sarsa(0.05), "Sarsa(0.05)"),
    (sarsa(0.05, lambda: Greedy(QSym)), "Sarsa(0.05)+Sym"),
#    (qlearn(0.05), "Qlearn(0.05)"),
#    (qlearn(0.05, lambda: Greedy(QSym)), "Qlearn(0.05)+Sym"),
]
plot(result, 'out.png', baseline=0.58)

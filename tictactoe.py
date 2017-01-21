"""
tic tac toe

0: vacant
1: first player
2: second player
"""
import numpy as np

def _board_to_int(board):
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
        #self.value = [0] * (3 ** 9 * 9)
        self.value = {}
    def get(self, info, a):
        return self.value.get(self.get_key(info, a), 0)
    def set(self, info, a, v):
        self.value[self.get_key(info, a)] = v
    def get_key(self, info, a):
        s = _board_to_int(info)
        return s * 9 + a

def array_to_int(xs, m, n):
    "convert array to [0, m ** n) integer"
    assert len(xs) == n
    s = 0
    for i in range(n):
        assert 0 <= xs[i] < m
        s += xs[i] * (m ** i)
    return s


class QSym(QRaw):
    "Compress states using symmetry"
    def get_key(self, info, a):
        board = info.copy()
        board[a] += 3  # where to place
        buf = []
        for i in range(4):
            buf.append(array_to_int(board, 6, 9))
            # rotate 90 ccw
            board = board[[2, 5, 8, 1, 4, 7, 0, 3, 6]]
        # mirror
        board = board[[2, 1, 0, 5, 4, 3, 8, 7, 6]]
        for i in range(4):
            buf.append(array_to_int(board, 6, 9))
            # rotate 90 ccw
            board = board[[2, 5, 8, 1, 4, 7, 0, 3, 6]]

        return min(buf)


class QLine1(QRaw):
    def to_lines(self, info, a):
        board = info.copy()
        board[a] += 3  # where to place
        lines = np.array([board[line] for line in LINES])
        lines.sort(axis=1)
        lines.sort(axis=0)
        return lines

    def get_key(self, info, a):
        lines =  self.to_lines(info, a)
        # list(sorted(set(tuple(sorted((a, b, c))) for a in range(6) for b in range(6) for c in range(6) if one(x > 2 for x in (a, b, c)))))
        MAP = [(0, 0, 3), (0, 0, 4), (0, 0, 5), (0, 1, 3),
               (0, 1, 4), (0, 1, 5), (0, 2, 3), (0, 2, 4),
               (0, 2, 5), (1, 1, 3), (1, 1, 4), (1, 1, 5),
               (1, 2, 3), (1, 2, 4), (1, 2, 5), (2, 2, 3),
               (2, 2, 4), (2, 2, 5)]
        MAP = dict((MAP[i], i + 1) for i in range(len(MAP)))
        lines = [MAP.get(tuple(line), 0) for line in lines]
        lines.sort()
        return tuple(lines)


class QLine2(QLine1):
    def get_key(self, info, a):
        lines = self.to_lines(info, a)
        MAP = {(2, 2, 3): 1, (1, 1, 3): 2,
               (0, 2, 3): 3, (0, 1, 3): 4,
               (0, 0, 3): 5
        }
        lines = [MAP.get(tuple(line), 0) for line in lines]
        lines.sort()
        print lines
        return tuple(lines)


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

    def get_info(self):
        return self.board.copy()

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
        self.Qtable = QClass()

    def __call__(self, env):
        from random import choice
        actions = get_available_actions(env)
        qa = [(self.Qtable.get(env.get_info(), a), a) for a in actions]
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



def sarsa(alpha, policyClass=Greedy, extra_play=False):
    global environment, policy
    gamma = 0.9
    num_result = batch_width * num_batch
    environment = Environment()
    policy = policyClass()
    action = policy(environment)
    prev_num_battle = 0
    win_ratios = []
    while True:
        old = environment.get_info()
        next_env, reward = environment(action)

        # determine a'
        next_action = policy(next_env)
        nextQ = policy.Qtable.get(next_env.get_info(), next_action)

        # update Q(s, a)
        Qsa = policy.Qtable.get(old, action)
        estimated_reward = reward + gamma * nextQ
        diff = estimated_reward - Qsa
        policy.Qtable.set(old, action, Qsa + alpha * diff)

        action = next_action

        num_battle = len(environment.result_log)
        if extra_play and num_battle > prev_num_battle:
            if num_battle % batch_width == 0:
                v = calc_win_ratio_from_plays(policy, policy_random)
                win_ratios.append(v)
        if num_battle == num_result:
            break
        prev_num_battle = num_battle

    if not extra_play:
        win_ratios = calc_win_ratios_from_result_log(environment)
    return win_ratios

def calc_win_ratio_from_plays(policy, op_policy=policy_random, N=1000):
    v = calc_win_ratio(
        [play(policy, op_policy) for i in range(1000)])
    return v

def calc_win_ratio(xs):
    c = Counter(xs)
    return float(c[1]) / len(xs)


def calc_win_ratios_from_result_log(env):
    vs = []
    for i in range(num_batch):
        v = calc_win_ratio(
            environment.result_log[
                batch_width * i :
                batch_width * (i + 1)])
        vs.append(v)
    return vs


def qlearn(alpha, policyClass=Greedy, extra_play=False):
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
        if extra_play and num_battle > prev_num_battle:
            if num_battle % batch_width == 0:
                v = calc_win_ratio_from_plays(policy, policy_random)
                win_ratios.append(v)
        if num_battle == num_result:
            break
        prev_num_battle = num_battle

    if not extra_play:
        win_ratios = calc_win_ratios_from_result_log(environment)
    return win_ratios


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


#batch_width = 10
#num_batch = 100
#result = [
#    (sarsa(0.05), "Sarsa(0.05)"),
#    (sarsa(0.05, lambda: Greedy(QSym)), "Sarsa(0.05)+Sym"),
#    (sarsa(0.05, lambda: Greedy(QLine1)), "Sarsa(0.05)+Line1"),
#    (sarsa(0.05, lambda: Greedy(QLine2)), "Sarsa(0.05)+Line2"),
#    (qlearn(0.05), "Qlearn(0.05)"),
#    (qlearn(0.05, lambda: Greedy(QSym)), "Qlearn(0.05)+Sym"),
#]
#plot(result, 'out.png', baseline=0.58)


batch_width = 300
num_batch = 1
def foo(f):
    win_ratios = []
    for i in range(100):
        f()
        v = calc_win_ratio_from_plays(policy, policy_random)
        win_ratios.append(v)
        print i, v
    s = np.array(win_ratios)
    print len(win_ratios)
    print "{:.2f}+-{:.2f}".format(s.mean(), s.std() * 2)

#foo(lambda: sarsa(0.05))
#foo(lambda: sarsa(0.05, lambda: Greedy(QSym)))
#foo(lambda: sarsa(0.05, lambda: Greedy(QLine1)))
foo(lambda: sarsa(0.05, lambda: Greedy(QLine2)))

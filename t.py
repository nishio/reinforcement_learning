"""
Quarto

0: vacant
1-16: occupied
"""
import numpy as np


def board_to_int(board):
    s = 0
    for i in range(16):
        s += board[i] * (17 ** i)
    return s

def board_to_possible_hands(board):
    return [i for i in range(16) if board[i] == 0]

def init_board():
    return np.zeros(16, dtype=np.int)

def init_Q():
    from scipy.sparse import dok_matrix
    return dok_matrix((17 ** 16, 16))

LINES = [
    [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
    [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15],
    [0, 5, 10, 15], [3, 6, 9, 12]
]
def is_win(board):
    for line in LINES:
        xs = board[line]
        if any(x == 0 for x in xs): continue
        a, b, c, d = xs - 1
        if a & b & c & d != 0:
            return 1
        if a | b | c | d != 15:
            return 1
    return 0

def print_board(board):
    """
    >>> print_board(range(16))
    . o x o | . o o x | . o o o | . o o o
    x o x o | x o o x | o x x x | o o o o
    x o x o | x o o x | x o o o | o x x x
    x o x o | x o o x | o x x x | x x x x
    """
    m = np.zeros((16, 4), dtype=np.int)
    for i in range(16):
        if board[i] == 0:
            m[i, :] = 0
        else:
            v = board[i] - 1
            for bit in range(4):  #  nth bit
                m[i, bit] = ((v >> bit) & 1) + 1

    for y in range(4):
        print ' | '.join(
            ' '.join(
                ['.ox'[v] for v in m[y * 4 : (y + 1) * 4, bit]]
            )
        for bit in range(4))
    print


def policy_random(env):
    from random import choice
    position = choice(board_to_possible_hands(env.board))
    piece = choice(env.available_pieces)
    return (position, piece)


class Environment(object):
    def __init__(self, policy=policy_random):
        self.board = init_board()
        self.available_pieces= range(2, 17)
        self.selected_piece = 1
        self.op_policy = policy
        self.result_log =[]

    def _update(self, action, k=1, to_print=False):
        position, piece = action

        if self.board[position] != 0:
            # illegal move
            print 'illegal pos'
            self.board = init_board()
            self.result_log.append(-1 * k)
            return (self.board, -1 * k)

        if piece not in self.available_pieces:
            # illegal move
            print 'illegal piece'
            self.board = init_board()
            self.result_log.append(-1 * k)
            return (self.board, -1 * k)

        self.board[position] = self.selected_piece
        self.available_pieces.remove(piece)
        self.selected_piece = piece

        if to_print:
            print k, action
            print_board(self.board)

        b = is_win(self.board)
        if b:
            self.board = init_board()
            self.result_log.append(+1 * k)
            return (self.board, +1 * k)

        if not self.available_pieces:
            # put selected piece
            self.board[self.board==0] = self.selected_piece
            b = is_win(self.board)
            if to_print:
                print 'last move'
                print_board(self.board)

            self.board = init_board()
            if b:
                # opponent win
                self.result_log.append(-1 * k)
                return (self.board, -1 * k)
            else:
                # tie
                self.result_log.append(0)
                return (self.board, -1)

        return None

    def __call__(self, action, to_print=False):
        ret = self._update(action, k=1, to_print=to_print)
        if ret: return ret
        op_action = self.op_policy(self)
        ret = self._update(op_action, k=-1, to_print=to_print)
        if ret: return ret
        return (self.board, 0)


def play(policy1, policy2=policy_random, to_print=False):
    env = Environment()
    result = 0

    for i in range(9):
        a = policy1(env)
        s, r = env(a, to_print=to_print)
        if r != 0: break
    if to_print:
        print env.result_log[-1]
    return env.result_log[-1]

#play(policy_random, to_print=True)

from collections import Counter
if 1:
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

def sarsa(alpha):
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
    return vs

if not'ex4':
    vs1 = sarsa(0.5)
    vs2 = sarsa(0.05)
    vs3 = sarsa(0.005)

    import matplotlib.pyplot as plt
    plt.plot([0.58] * len(vs1), label = "baseline")
    plt.plot(vs1, label = "Sarsa(0.5)")
    plt.plot(vs2, label = "Sarsa(0.05)")
    plt.plot(vs3, label = "Sarsa(0.005)")
    plt.xlabel("iteration")
    plt.ylabel("Prob. of win")
    plt.legend(loc = 4)
    plt.savefig('sarsa.png')

def f(n, m):
    if m == 1: return n + 1
    return n * f(n - 1, m - 1) + f(n, m - 1)


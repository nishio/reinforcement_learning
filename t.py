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


board = init_board()
print_board(board)

for i in range(9):
    color = 1 if i % 2 == 0 else 2
    a = policy_random(board)
    board[a] = color
    print_board(board)
    b = is_win(board)
    if b:
        print b
        break

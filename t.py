from random import random
import numpy as np

def action_50():
    if random() < 0.1:
        return 500
    return 0

def action_100():
    if random() < 0.2:
        return 500
    return 0

def action_150():
    if random() < 0.3:
        return 500
    return 0

def action_none():
    return 100

actions = [action_50, action_100, action_150, action_none]
num_actions = len(actions)

def policy_greedy(num_use, sum_reward):
    for i in range(num_actions):
        if num_use[i] < 3:
            return i
    return np.argmax(sum_reward / num_use)


def policy_builder_greedy(threshold):
    def policy_greedy(num_use, sum_reward):
        for i in range(num_actions):
            if num_use[i] < threshold:
                return i
        return np.argmax(sum_reward / num_use)
    return policy_greedy


def ex1():
    for i in range(20):
        i_action = policy_greedy(num_use, sum_reward)
        action = actions[i_action]
        reward = action()
        print action.__name__, reward
        num_use[i_action] += 1
        sum_reward[i_action] += reward


NUM_SERIES = 600
NUM_ITERATION = 1000
def ex2(name, policy):
    choices = np.zeros((NUM_SERIES, NUM_ITERATION))
    rewards = np.zeros((NUM_SERIES, NUM_ITERATION))
    for j in range(NUM_SERIES):
        num_use = np.zeros(num_actions)
        sum_reward = np.zeros(num_actions)
        for i in range(NUM_ITERATION):
            i_action = policy(num_use, sum_reward)
            action = actions[i_action]
            reward = action()
            #print action.__name__, reward
            num_use[i_action] += 1
            sum_reward[i_action] += reward
            choices[j, i] = i_action
            rewards[j, i] = reward


    last = np.sort(choices[:, -1])
    print name
    print sum(last == 0)
    print sum(last == 1)
    print sum(last == 2)
    print sum(last == 3)

    # visualization
    import matplotlib.pyplot as plt
    choices.sort(axis=0)
    plt.imshow(choices)
    plt.savefig('{}.png'.format(name))

ex2("greedy_1", policy_builder_greedy(1))
ex2("greedy_3", policy_builder_greedy(3))
ex2("greedy_10", policy_builder_greedy(10))

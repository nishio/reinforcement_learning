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


def policy_builder_optimistic(num_offset, offset_weight=500):
    def policy_optimistic(num_use, sum_reward):
        return np.argmax(
            (sum_reward + num_offset * offset_weight)
            / (num_use + num_offset))
    return policy_optimistic


def policy_ucb1(num_use, sum_reward):
    for i in range(num_actions):
        if num_use[i] < 1:
            return i
    mu = sum_reward / num_use
    ub = 500 * np.sqrt(np.log(num_use.sum()) * 2 / num_use)
    return np.argmax(mu + ub)


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
def ex2(name, policy, num_iteration=NUM_ITERATION):
    choices = np.zeros((NUM_SERIES, num_iteration))
    rewards = np.zeros((NUM_SERIES, num_iteration))
    for j in range(NUM_SERIES):
        num_use = np.zeros(num_actions)
        sum_reward = np.zeros(num_actions)
        for i in range(num_iteration):
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
    if num_iteration == 10000:
        plt.imshow(choices[:, ::10])
    else:
        plt.imshow(choices)
    plt.savefig('{}.png'.format(name))


def run_ex2():
    ex2("greedy_1", policy_builder_greedy(1))
    ex2("greedy_3", policy_builder_greedy(3))
    ex2("greedy_10", policy_builder_greedy(10))

    ex2("ucb1", policy_ucb1)

    ex2("optimistic_1", policy_builder_optimistic(1))
    ex2("optimistic_3", policy_builder_optimistic(3))
    ex2("optimistic_10", policy_builder_optimistic(10))

    ex2("ucb1_long", policy_ucb1, num_iteration=10000)


def ex3(name, policy, num_iteration=NUM_ITERATION):
    choices = np.zeros((NUM_SERIES, num_iteration))
    rewards = np.zeros((NUM_SERIES, num_iteration))
    for j in range(NUM_SERIES):
        num_use = np.zeros(num_actions)
        sum_reward = np.zeros(num_actions)
        for i in range(num_iteration):
            i_action = policy(num_use, sum_reward)
            action = actions[i_action]
            reward = action()
            #print action.__name__, reward
            num_use[i_action] += 1
            sum_reward[i_action] += reward
            choices[j, i] = i_action
            rewards[j, i] = reward

    return rewards.mean(axis=0)
    # visualization
    #import matplotlib.pyplot as plt
    #plt.plot(rewards.sum(axis=0))
    #plt.savefig('{}_reward.png'.format(name))


def ex3_vis(p=100, N=1000):
    r1 = ex3(
        "greedy_{}".format(p),
        policy_builder_greedy(p), N)
    r2 = ex3(
        "optimistic_{}".format(p),
        policy_builder_optimistic(p), N)
    r3 = ex3("ucb1", policy_ucb1, N)

    if N == 1000:
        w = 30
    elif N == 20000:
        w = 300
    else:
        raise NotImplemented

    kernel = np.ones(w) / w
    def smooth(x):
        return np.convolve(x, kernel, mode='valid')

    import matplotlib.pyplot as plt
    plt.plot(smooth(r1), label = "greedy_{}".format(p))
    plt.plot(smooth(r2), label = "optimistic_{}".format(p))
    plt.plot(smooth(r3), label = "ucb1")

    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.legend(loc = 4)

    plt.savefig('rewards_{}_3.png'.format(N))

#ex3_vis(N=20000)

if not"ex2-5":
    ex2("greedy_100", policy_builder_greedy(100))
    ex2("optimistic_100", policy_builder_optimistic(100))

actions = [action_none, action_50, action_100, action_150]
ex2("greedy_3_reorder", policy_builder_greedy(3))

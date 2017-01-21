import numpy as np
from collections import Counter
WORLD_WIDTH = 300
WORLD_HEIGHT = 200
START_X = 100.0
START_Y = 100.0

class Environment(object):
    def __init__(self):
        self.result_log =[]
        self.init_state()

    def init_state(self):
        self.state = np.array([
            START_X, START_Y, 0.0, 0.0])
        self.time = 300

    def update(self, action):
        self.state[2:] += action
        self.state[:2] += self.state[2:]
        x, y = self.state[:2]

        if x < 0:
            self.init_state()
            self.result_log.append('left')
            return -1

        if y < 0:
            self.init_state()
            self.result_log.append('top')
            return -1

        if y > WORLD_HEIGHT:
            self.init_state()
            self.result_log.append('bottom')
            return -1

        if x > WORLD_WIDTH:
            self.init_state()
            self.result_log.append('goal')
            return 1

        self.time -= 1
        if self.time == 0:
            self.init_state()
            self.result_log.append('timeout')
            return -1

        return 0

def policy_random(env):
    return np.random.normal(size=2)

def play(policy, num_plays=100, to_print=False):
    env = Environment()
    result = 0
    for i in range(num_plays):
        while True:
            a = policy(env)
            r = env.update(a)
            if r:
                break
        #print env.result_log[-1]
    return env

print Counter(play(policy_random).result_log)

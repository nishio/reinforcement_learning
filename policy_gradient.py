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
            START_X, START_Y,
            np.random.normal(0, 0.1), np.random.normal(0, 0.1),
            1.0])
        self.time = 300

    def get_state(self):
        return self.state.copy()

    def update(self, action):
        m = np.linalg.norm(action)
        if m > 1:
            action /= m
        self.state[2:4] += action / 30
        #m = np.linalg.norm(self.state[2:])
        #if m > 10:
        #    self.state[2:] /= (m / 10)

        self.state[:2] += self.state[2:4]
        x, y = self.state[:2]

        if x < 0:
            self.init_state()
            self.result_log.append('left')
            return -1.0

        if y < 0:
            self.init_state()
            self.result_log.append('top')
            return -1.0

        if y > WORLD_HEIGHT:
            self.init_state()
            self.result_log.append('bottom')
            return -1.0

        if x > WORLD_WIDTH:
            self.init_state()
            self.result_log.append('goal')
            return 10.0

        self.time -= 1
        if self.time == 0:
            self.init_state()
            self.result_log.append('timeout')
            return -1.0

        return 0.0

def policy_random(state):
    return np.random.normal(size=2)


class Policy(object):
    def __init__(self):
        self.theta = np.random.normal(scale=0.001, size=(5, 2))

    def __call__(self, state):
        mean = state.dot(self.theta)
        a = np.random.normal(mean)
        return a

    def grad(self, state, action):
        t1 = action - state.dot(self.theta)
        # 2
        t2 = -state
        # 4
        g = np.outer(t2, t1)
        return g


def play(policy, num_plays=100, to_print=False):
    env = Environment()
    result = 0
    for i in range(num_plays):
        while True:
            s = env.get_state()
            a = policy(s)
            r = env.update(a)
            if r:
                break
        #print env.result_log[-1]
    return env


def reinforce(policy, num_plays=100, to_print=False):
    env = Environment()
    result = 0
    samples = []
    sum_t = 0
    sum_r = 0.0
    for i in range(num_plays):
        t = 0
        SARs = []
        while True:
            s = env.get_state()
            a = policy(s)
            r = env.update(a)

            t += 1
            sum_r += r
            SARs.append((s, a, r))
            if r:
                break
        samples.append((t, SARs))
        sum_t += t

    baseline = float(sum_r) / sum_t
    grad = np.zeros((5, 2))
    for (t, SARs) in samples:
        tmp_grad = np.zeros((5, 2))
        for (s, a, r) in SARs:
            g = policy.grad(s, a)
            tmp_grad += g * (r - baseline)
        grad += tmp_grad / t
    grad /= num_plays
    #policy.theta /= np.linalg.norm(policy.theta)
    if np.linalg.norm(grad) > 1:
        grad /= np.linalg.norm(grad)
    print 'theta'
    print policy.theta
    print 'grad'
    print grad
    policy.theta += 0.01 * grad
    print baseline, sum_t
    return env, samples


#print Counter(play(policy_random).result_log)
#print Counter(play(Policy()).result_log)
policy = Policy()
for i in range(100):
    env, samples = reinforce(policy, 100)
    print Counter(env.result_log)

    from PIL import Image, ImageDraw
    im = Image.new('RGB', (300, 200), color=(255,255,255))
    d = ImageDraw.Draw(im)
    for t, SARs in samples:
        points = [(100, 100)]
        for s, a, r in SARs:
            points.append(tuple(s[:2]))
        d.line(points, fill=0)
    im.save('reinforce{:02d}.png'.format(i))



"""
POMDP
"""
import numpy as np
from matplotlib import pyplot as plt

actions = ['Aleft', 'Aright', 'Alisten']
states = ['Sleft', 'Sright']
observations = ['Oleft', 'Oright']

# R(s, a)
reward = np.array([
    [+10.0, -100.0, -1.0],
    [-100.0, +10.0, -1.0],
])

# T(s, a, s')
transition = np.array([
    [[0.5, 0.5], [0.5, 0.5],[1.0, 0.0]],
    [[0.5, 0.5], [0.5, 0.5],[0.0, 1.0]]
])

# O(s', a, o)
observation = np.array([
    [[0.5, 0.5], [0.5, 0.5],[0.85, 0.15]],
    [[0.5, 0.5], [0.5, 0.5],[0.15, 0.85]]
])

# initial beleif
b0 = np.array([0.5, 0.5])

V0 = np.array([[0.0, 0.0]])

gamma = 1.0

def _common_part(V0):
    g_a_star = reward
    g_a_o = np.zeros((len(actions), len(observations), len(V0), len(states)))
    for i in range(len(V0)):
        for a in range(len(actions)):
            for o in range(len(observations)):
                buf = np.zeros(len(states))
                for s in range(len(states)):
                    sum_s2 = 0.0
                    for s2 in range(len(states)):
                        sum_s2 += transition[s, a, s2] * observation[s2, a, o] * V0[i][s2]
                    buf[s] = gamma * sum_s2
                g_a_o[a, o, i, :] = buf
    return g_a_star, g_a_o


def exact_backup(V0):
    g_a_star, g_a_o = _common_part(V0)
    #print 'g_a_o'
    #print g_a_o
    # exact
    tmp = g_a_star.copy()
    g_a = np.zeros((len(actions), len(V0) ** len(observations), len(states)))
    for a in range(len(actions)):
        #print tmp[:, a]
        lhs = [tmp[:, a]]

        for o in range(len(observations)):
            #print g_a_o[a, o]
            rhs= g_a_o[a, o]
            buf = []
            #print "lhs: {}, rhs: {}".format(lhs, rhs)
            for x in lhs:
                for y in rhs:
                    buf.append(x + y)
            lhs = buf
            #print lhs
        g_a[a,:] = lhs
    #print g_a
    V1 = g_a.reshape(len(actions) * len(V0) ** len(observations), len(states))
    return V1


def pbvi_backup(V0, B):
    g_a_star, g_a_o = _common_part(V0)
    g_a_b = np.zeros((len(actions), len(B), len(states)))
    for a in range(len(actions)):
        for b in range(len(B)):
            tmp = g_a_star[:, a].copy()
            for o in range(len(observations)):
                i = np.argmax([alpha.dot(B[b]) for alpha in g_a_o[a, o]])
                tmp += g_a_o[a, o, i]
            g_a_b[a, b] = tmp

    V1 = np.array([
        g_a_b[np.argmax([g_a_b[a, b].dot(B[b]) for a in range(len(actions))]), b]
        for b in range(len(B))
    ])
    return V1


def plot(V):
    for alpha in V:
        plt.plot(alpha)
    plt.show()


def prune(V):
    ret = []
    N = 10000
    for i in range(N + 1):
        s = float(i) / N
        b = [s, 1 - s]
        ret.append(np.argmax([x.dot(b) for x in V]))
    #print ret
    return V[list(set(ret))]


def get_max(V):
    vs = []
    N = 10000
    for i in range(N + 1):
        s = float(i) / N
        b = [s, 1 - s]
        vs.append(np.max([x.dot(b) for x in V]))
    return vs



def update_belief(b, a, o):
    ret = np.zeros(len(states))
    for s2 in range(len(states)):
        tmp = np.sum([b[s] * transition[s, a, s2] for s in range(len(states))])
        ret[s2] = tmp * observation[s2, a, o]
    z = ret.sum()
    return ret / z


from random import random
def belief_point_set_expansion(B0):
    print
    ret = list(B0)
    for b in B0:
        buf = []
        s = 1 if random() < b[1] else 0
        for a in range(len(actions)):
            p = transition[s, a][1]
            s2 = 1 if random() < p else 0
            p = observation[s2, a][1]
            o = 1 if random() < p else 0

            b2 = update_belief(b, a, o)
            buf.append(b2)
        dists = [np.min([np.linalg.norm(x - y) for y in B0]) for x in buf]
        print b
        print dists
        i = np.argmax(dists)
        if dists[i] > 0.0:
            ret.append(buf[i])

    return ret


def get_B():
    b = [np.array([0.5, 0.5])]
    while True:
        b = belief_point_set_expansion(b)
        print b
        if len(b) > 10: break
    return b


if 0:
    N = 10
    B = []
    for i in range(N + 1):
        s = float(i) / N
        b = [s, 1 - s]
        B.append(b)
    B = np.array(B)
else:
    B = get_B()

if 'pbvi':
    backup = lambda V: pbvi_backup(V, B)
else:
    backup = exact_backup

if 0:
    from kagura import stopwatch
    s = stopwatch.Stopwatch(quiet=True)
    Vprev = V0
    for i in range(10):
        s.start()
        V = backup(Vprev)
        beforePrune = len(V)
        V = prune(V)
        print "V{}, {}->{}, {}".format(i + 1, beforePrune, len(V), s.get())
        s.end()
        Vprev = V


import kagura
if 1:
    N = 10000
    xs = [float(i) / N for i in range(N + 1)]
    plt.plot(xs, get_max(kagura.load('exactVI')), label='exact')
    plt.plot(xs, get_max(kagura.load('PBVI')), label='PBVI')
    plt.plot(xs, get_max(kagura.load('PBVI2')), label='PBVI2')
    V = kagura.load('PBVI2')
    plt.scatter(np.array(B)[:, 1], [np.max([b.dot(v) for v in V]) for b in B])
    plt.legend(loc = 4)
    plt.xlim(0, 1)
    plt.show()

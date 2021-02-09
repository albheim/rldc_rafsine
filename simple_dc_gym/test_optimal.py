from ray import tune
import numpy as np

from dc import SimpleDCEnv

dc = SimpleDCEnv(20)

state = dc.reset()

hist = []

for i in range(10000):
    a = np.random.randint(dc.n_servers)
    flow = np.sum(state["flow"])
    temp = 23
    state, reward = dc.step((a, [temp, flow]))
    hist.append(reward)
    l = min(len(hist), 1000)
    print(min(hist[-l:]), max(hist[-l:]))

#-4.6 to -3.6
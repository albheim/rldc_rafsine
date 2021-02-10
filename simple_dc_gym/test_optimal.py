import numpy as np

from dc import SimpleDCEnv

dc = SimpleDCEnv({"n_servers": 20})

state = dc.reset()

hist = []

for i in range(10000):
    place = np.random.randint(dc.n_servers)
    state = (state[:-1] * (dc.shigh - dc.slow) + dc.shigh + dc.slow) / 2
    flow = np.sum(state[20:40])
    temp = 23
    crah = np.array([temp, flow])
    crah = (2*crah - dc.alow - dc.ahigh) / (dc.ahigh - dc.alow)
    state, reward, _, _ = dc.step((place, crah))
    hist.append(reward)
    l = min(len(hist), 1000)
    print(min(hist[-l:]), max(hist[-l:]))

#-4.6 to -3.6
#-4.5 to -0.1
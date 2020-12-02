import datetime

import numpy as np
import tensorflow as tf

from dc import SimpleDCEnv
from rl import DDPG

dc = SimpleDCEnv()

agent = DDPG(dc.n_servers)

tag = "test_nan_only_cpu_cost"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + tag + "_" + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

prev_state = dc.reset()
prev_state_arr = np.hstack(
    (prev_state["load"], prev_state["out_temp"], prev_state["job"], prev_state["ambient"]))

total_episodes = 1000000

for ep in range(total_episodes):

    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_arr), 0)

    action = agent.policy(tf_prev_state)
    a = (np.random.choice(dc.n_servers,
                          p=action[:dc.n_servers]), action[-2], action[-1])
    # Recieve state and reward from environment.
    #crah_flow = np.sum(prev_state["flow"])
    #a = [a, (17 - 10) / 15, (crah_flow - dc.crah_min_flow) / (dc.crah_max_flow - dc.crah_min_flow)]
    state, reward = dc.step(a)

    #hist.append((state, a, reward))

    with summary_writer.as_default():
        tf.summary.scalar("value/reward", reward, step=ep)
        tf.summary.scalar("value/cost", state["cost"], step=ep)
        tf.summary.scalar("temperature/ambient", state["ambient"], step=ep)
        tf.summary.scalar("temperature/serverin", state["in_temp"], step=ep)
        tf.summary.scalar("temperature/crahout",
                          state["crah_temp_out"], step=ep)
        tf.summary.scalar("job", state["job"], step=ep)
        tf.summary.scalar("generator", state["generator"], step=ep)
        #tf.summary.scalar("flow/water", state["flow_main"][0], step=ep)
        tf.summary.scalar("flow/crah", state["crah_flow"], step=ep)
        tf.summary.scalar("flow/servertot", np.sum(state["flow"]), step=ep)
        tf.summary.scalar("generator", state["generator"], step=ep)
        for i in range(dc.n_servers):
            tf.summary.scalar("server{}/flow".format(i),
                              state["flow"][i], step=ep)
            tf.summary.scalar("server{}/tempcpu".format(i),
                              state["out_temp"][i], step=ep)
            tf.summary.scalar("server{}/tempout".format(i),
                              state["cpu_temp"][i], step=ep)
            tf.summary.scalar("server{}/load".format(i),
                              state["load"][i], step=ep)

    state_arr = np.hstack(
        (state["load"], state["out_temp"], state["job"], state["ambient"]))

    agent.buffer.record((prev_state_arr, action, reward, state_arr))

    agent.learn()
    agent.update_targets()

    prev_state_arr = state_arr
    prev_state = state

    print("{}/{} - rew {} - act {}".format(ep, total_episodes, reward, a))

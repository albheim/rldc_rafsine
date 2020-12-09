import datetime

import numpy as np
import tensorflow as tf

from dc import SimpleDCEnv
from rl import DDPG

dc = SimpleDCEnv(n_servers=20)

agent = DDPG(2 * dc.n_servers + 2, dc.n_servers + 2)

tag = "new_env"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + tag + "_" + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

prev_state = dc.reset()
prev_state_arr = np.hstack((prev_state["load"], prev_state["out_temp"], prev_state["jobs"], prev_state["ambient"]))

total_episodes = 1000000

for ep in range(total_episodes):

    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_arr), 0)

    action = agent.policy(tf_prev_state)
    #action[-2] = 0.8
    a = (action[:dc.n_servers], action[-2], action[-1])
    # Recieve state and reward from environment.
    #crah_flow = np.sum(prev_state["flow"])
    #a = [a, (17 - 10) / 15, (crah_flow - dc.crah_min_flow) / (dc.crah_max_flow - dc.crah_min_flow)]
    state, reward = dc.step(a)

    #hist.append((state, a, reward))

    with summary_writer.as_default():
        tf.summary.scalar("value/reward", reward, step=ep)
        tf.summary.scalar("value/cost", state["cost"], step=ep)
        tf.summary.scalar("temperature/ambient", state["ambient"], step=ep)
        tf.summary.scalar("temperature/crahout",
                          state["crah_temp_out"], step=ep)
        tf.summary.scalar("jobs", state["jobs"], step=ep)
        tf.summary.scalar("energy/compressor", state["energy_compressor"], step=ep)
        tf.summary.scalar("energy/server_fan", state["energy_server_fans"], step=ep)
        tf.summary.scalar("energy/crah_fan", state["energy_crah_fans"], step=ep)
        tf.summary.scalar("flow/crah", state["crah_flow"], step=ep)
        tf.summary.scalar("flow/servertot", np.sum(state["flow"]), step=ep)
        for i in range(dc.n_servers):
            tf.summary.scalar("server{}/flow".format(i),
                              state["flow"][i], step=ep)
            tf.summary.scalar("server{}/tempcpu".format(i),
                              state["out_temp"][i], step=ep)
            tf.summary.scalar("server{}/tempin".format(i), 
                              state["in_temp"][i], step=ep)
            tf.summary.scalar("server{}/tempout".format(i),
                              state["cpu_temp"][i], step=ep)
            tf.summary.scalar("server{}/load".format(i),
                              state["load"][i], step=ep)

    state_arr = np.hstack(
        (state["load"], state["out_temp"], state["jobs"], state["ambient"]))

    agent.buffer.record((prev_state_arr, action, reward, state_arr))

    agent.learn()
    agent.update_targets()

    prev_state_arr = state_arr
    prev_state = state

    print("{}/{} - rew {} - act {}".format(ep, total_episodes, reward, a))

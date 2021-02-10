from typing import Dict
import argparse
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        pass
        # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        # episode.user_data["states"] = [] # Save state

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        pass
        # env = base_env.get_unwrapped()[0]

        # s = episode.last_observation_for()

        # sfix = ((env.shigh - env.slow) * s + env.slow + env.shigh) / 2

        # episode.user_data["states"].append(sfix)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        env = base_env.get_unwrapped()[0]
        print(env.time)

        s = episode.last_observation_for()
        # job = s[-1]
        s = s[:-1]
        s = ((env.shigh - env.slow) * s + env.slow + env.shigh) / 2

        nstate = len(s) // 3
        for i in range(nstate):
            episode.custom_metrics[f"srv{i}/temp_out"] = s[i] 
            episode.custom_metrics[f"srv{i}/flow"] = s[nstate + i] 
            episode.custom_metrics[f"srv{i}/load"] = s[2 * nstate + i] 

            episode.custom_metrics[f"srv{i}/temp_cpu"] = env.server_temp_cpu[i]

        
        episode.custom_metrics[f"crah/temp_out"] = env.action[1][0]
        episode.custom_metrics[f"crah/flow"] = env.action[1][1]
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass
        # print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass
        # print("trainer.train() result: {} -> {} episodes".format(trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        # result["callback_ok"] = True

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        # if "num_batches" not in episode.custom_metrics:
        #     episode.custom_metrics["num_batches"] = 0
        # episode.custom_metrics["num_batches"] += 1
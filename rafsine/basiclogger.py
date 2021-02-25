from typing import Dict

import numpy as np

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class LoggingCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        pass
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass
    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        pass
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        env = base_env.get_unwrapped()[0]

        # To allow all vars to init
        if env.get_time() == 0:
            return

        print("Logging at env time {}".format(env.get_time()))

        # Log server
        for i in range(env.n_servers):
            episode.custom_metrics[f"srv{i}/load"] = env.server_load[i]
            episode.custom_metrics[f"srv{i}/flow"] = env.server_flow[i]
            episode.custom_metrics[f"srv{i}/temp_in"] = env.server_temp_in[i]
            episode.custom_metrics[f"srv{i}/temp_out"] = env.server_temp_out[i]
            episode.custom_metrics[f"srv{i}/temp_cpu"] = env.server_temp_cpu[i]

        for i in range(len(env.crah)):
            episode.custom_metrics[f"crah{i}/temp_in"] = env.crah_temp_in[i]
            episode.custom_metrics[f"crah{i}/temp_out"] = env.crah_temp_out[i]
            episode.custom_metrics[f"crah{i}/flow"] = env.crah_flow[i]

        episode.custom_metrics[f"job/dur"] = env.job[0]
        episode.custom_metrics[f"job/load"] = env.job[1]

        # Should be 0 with the drop instead of delay?
        episode.custom_metrics[f"job/queue"] = len(env.event_queue)

        episode.custom_metrics[f"other/ambient_temp"] = env.ambient_temp
        episode.custom_metrics[f"other/server_flow"] = np.sum(env.server_flow)

        episode.custom_metrics[f"power/crah_fan"] = env.crah_fan_power
        episode.custom_metrics[f"power/server_fan"] = env.server_fan_power
        episode.custom_metrics[f"power/compressor"] = env.compressor_power

        episode.custom_metrics[f"cost/energy"] = env.total_energy_cost
        episode.custom_metrics[f"cost/delay"] = env.total_job_drop_cost
        
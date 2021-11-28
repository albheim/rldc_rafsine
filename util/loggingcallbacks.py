from typing import Dict

import numpy as np

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class LoggingCallbacks(DefaultCallbacks):
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
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        pass
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        pass
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        env = base_env.get_unwrapped()[0].unwrapped

        # Just to be consistent with earlier logging
        worker_index = worker.worker_index - 1
        if worker_index != 0: # Only log single env for ppo
            return
        
        # Log server, this takes a lot of memory so turned off most of the time
        if env.loglevel > 1:
            # Does this do what I want?
            episode.hist_data[f"env{worker_index}/loads"] = env.servers.load.tolist()
            # episode.hist_data[f"env{worker_index}/cpu_temps"] = env.servers.temp_cpu.tolist()
            # episode.hist_data[f"env{worker_index}/flows"] = env.servers.flow.tolist()
            # episode.hist_data[f"env{worker_index}/inlet_temps"] = env.flowsim.server_temp_in.tolist()
            # episode.hist_data[f"env{worker_index}/outlet_temps"] = env.flowsim.server_temp_out.tolist()
            # What I really want is more like
            # episode.hist_data[f"env{worker_index}/loads"] = [i for (i, load) in enumerate(env.servers.load) for _ in range(load)]

            for i in range(env.n_servers):
                episode.custom_metrics[f"env{worker_index}/srv{i}/load"] = env.servers.load[i]
                episode.custom_metrics[f"env{worker_index}/srv{i}/temp_in"] = env.flowsim.server_temp_in[i]
                episode.custom_metrics[f"env{worker_index}/srv{i}/flow"] = env.servers.flow[i]
            #     episode.custom_metrics[f"env{worker_index}/srv{i}/temp_cpu"] = env.servers.temp_cpu[i]
            #     episode.custom_metrics[f"env{worker_index}/srv{i}/temp_out"] = env.flowsim.server_temp_out[i]

        if env.loglevel > 0:
            episode.custom_metrics[f"env{worker_index}/srv/max_temp_cpu"] = env.servers.temp_cpu.max()
            total_server_flow = np.sum(env.servers.flow)
            episode.custom_metrics[f"env{worker_index}/srv/server_total_flow"] = total_server_flow
            episode.custom_metrics[f"env{worker_index}/srv/overheated_inlets"] = env.servers.overheated_inlets
            episode.custom_metrics[f"env{worker_index}/srv/avg_temp_in"] = np.dot(env.flowsim.server_temp_in, env.servers.flow) / total_server_flow
            episode.custom_metrics[f"env{worker_index}/srv/min_temp_in"] = np.min(env.flowsim.server_temp_in)
            episode.custom_metrics[f"env{worker_index}/srv/max_temp_in"] = np.max(env.flowsim.server_temp_in)
            episode.custom_metrics[f"env{worker_index}/srv/avg_temp_out"] = np.dot(env.flowsim.server_temp_out, env.servers.flow) / total_server_flow
            episode.custom_metrics[f"env{worker_index}/srv/avg_temp_cpu"] = np.mean(env.servers.temp_cpu)
            episode.custom_metrics[f"env{worker_index}/srv/load_variance"] = np.var(env.servers.load)


            for i in range(env.n_crah):
                episode.custom_metrics[f"env{worker_index}/crah{i}/temp_in"] = env.flowsim.crah_temp_in[i]
                episode.custom_metrics[f"env{worker_index}/crah{i}/temp_out"] = env.crah.temp_out[i]
                episode.custom_metrics[f"env{worker_index}/crah{i}/flow"] = env.crah.flow[i]

            episode.custom_metrics[f"env{worker_index}/crah/crah_total_flow"] = np.sum(env.crah.flow)

            episode.custom_metrics[f"env{worker_index}/job/load"] = env.job[0]
            episode.custom_metrics[f"env{worker_index}/job/duration"] = env.job[1]

            episode.custom_metrics[f"env{worker_index}/job/running"] = len(env.servers.running_jobs)
            episode.custom_metrics[f"env{worker_index}/job/misplaced"] = env.servers.misplaced_jobs

            episode.custom_metrics[f"env{worker_index}/power/server_fan"] = env.servers.fan_power
            episode.custom_metrics[f"env{worker_index}/power/crah_fan"] = env.crah.fan_power
            episode.custom_metrics[f"env{worker_index}/power/compressor"] = env.crah.compressor_power
            it_power = np.sum(env.servers.load) + np.sum(env.servers.fan_power)
            cooling_power = env.servers.fan_power + env.crah.fan_power + env.crah.compressor_power
            episode.custom_metrics[f"env{worker_index}/power/total_server_load"] = it_power
            episode.custom_metrics[f"env{worker_index}/power/PUE"] = (cooling_power + it_power) / it_power

            episode.custom_metrics[f"env{worker_index}/cost/energy"] = env.total_energy_cost
            episode.custom_metrics[f"env{worker_index}/cost/misplaced"] = env.total_job_misplace_cost
            episode.custom_metrics[f"env{worker_index}/cost/temp_cold_isle"] = env.total_overheat_cost
            episode.custom_metrics[f"env{worker_index}/cost/load_variance"] = env.total_load_variance_cost
            episode.custom_metrics[f"env{worker_index}/cost/total"] = env.total_energy_cost + env.total_job_misplace_cost + env.total_overheat_cost + env.total_load_variance_cost
            
            episode.custom_metrics[f"env{worker_index}/other/outdoor_temp"] = env.outdoor_temp(env.time)

            
import os
from datetime import datetime
import numpy as np

import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog

from util.parsing import parse_all
from util.loggingcallbacks import LoggingCallbacks 
from dc.dc import DCEnv
from models.serverconv import ServerConvNetwork
from models.serveronlyconv import ServerOnlyConvNetwork
from models.serverconv2d import ServerConv2DNetwork
from models.crahonly import CRAHOnlyNetwork
from models.emptynet import EmptyNetwork

def run(
    rafsine = False,
    crah_out_setpoint = 22.0,
    crah_flow_setpoint = 0.8,
    n_bins = 0,
    break_after = -1.0,

    avg_load = 100.0,
    job_p = 0.5,

    model = "baseline",
    alg = "PPO",
    seed = 37,
    tag = "default",
    n_envs = 1,
    timesteps = 500000,
    n_samples = 1,
    horizon = 200,
    output = None,
    hyperopt = False,

    loglevel = 1,
    verbose = 0,
    restore = None,
):
    # Register env with ray
    tune.register_env("DCEnv", DCEnv)

    # Register model with ray
    if model == "serverconv":
        ModelCatalog.register_custom_model("serverconv", ServerConvNetwork)
        actions = ["server", "crah_out", "crah_flow"]
        observations = ["temp_out", "load", "outdoor_temp", "job"]
    elif model == "serveronlyconv":
        ModelCatalog.register_custom_model("serveronlyconv", ServerOnlyConvNetwork)
        actions = ["server"]
        observations = ["temp_out", "load", "job"]
    elif model == "serverconv2d":
        ModelCatalog.register_custom_model("serverconv2d", ServerConv2DNetwork)
        actions = ["server", "crah_out", "crah_flow"]
        observations = ["temp_out", "load", "outdoor_temp", "job"]
    elif model == "crahonly":
        ModelCatalog.register_custom_model("crahonly", CRAHOnlyNetwork)
        actions = ["crah_out", "crah_flow"]
        observations = ["temp_out", "load", "outdoor_temp"]
    elif model == "baseline":
        ModelCatalog.register_custom_model("baseline", EmptyNetwork)
        actions = ["none"]
        observations = ["temp_out", "load", "outdoor_temp", "job"]

    # Some common config
    tune_config = {
        # Environment
        "env": "DCEnv",
        "env_config": {
            "rafsine_flow": rafsine,
            "seed": seed,
            "actions": actions,
            "observations": observations,
            "crah_out_setpoint": crah_out_setpoint,
            "crah_flow_setpoint": crah_flow_setpoint,
            "avg_load": avg_load,
            "loglevel": loglevel,
            "n_bins": n_bins,
            "break_after": break_after,
            "job_p": job_p,
        },

        "model": {
            "custom_model": model,
            "custom_model_config": {
                "n_servers": 360,
                "activation": "elu", 
                "n_hidden": 64, 
                "rack_inject": True, 
                "conv_filter_size": 11, 
                "n_conv_layers": 1, 
                "n_conv_hidden": 3,
                "n_crah_layers": 1,
                "n_value_layers": 2,
                "crah_input": "other",
                "value_input": "all",
            },
            "use_lstm": False,
        },

        # Agent settings
        "vf_clip_param": 1000.0,
        "entropy_coeff": 0,
        "kl_target": 0.01,
        "clip_param": 0.3,

        # Worker setup
        "num_workers": n_envs, # How many workers are spawned, ppo use this many envs and data is aggregated from all
        "num_envs_per_worker": 1, # How many envs on each worker? Can be used to vectorize, probably same as num_workers?
        "num_gpus_per_worker": 1 if rafsine else 0, # Only give gpu to rafsine
        "num_cpus_per_worker": 1, # Does this make any difference?
        "seed": seed,

        # For logging (does soft_horizon do more, not sure...)
        "callbacks": LoggingCallbacks,
        "soft_horizon": True,
        "no_done_at_end": True,
        "horizon": horizon, # Decides length of episodes for logdata collection, should for now be same as rollout_frament_length
        "output": output,

        # Training
        # "train_batch_size": n_envs * horizon, # Collects batch of data from different workes, does min/max/avg over it and trains on it
        "train_batch_size": n_envs * horizon, # Collects batch of data from different workes, does min/max/avg over it and trains on it
        "rollout_fragment_length": horizon, # How much data is colelcted by each worker before sending in data for training
        "metrics_smoothing_episodes": 1, # rolling avg

        # Data settings
        #"observation_filter": "MeanStdFilter", # Test this
        #"normalize_actions": True,

        # Evaluation
        # "evaluation_num_workers": 1,
        # "evaluation_interval": 10, # Every n times we train we also evaluate
        # "evaluation_num_episodes": 24 * 3600 // horizon, # One day
        # "evaluation_config": {
        #     "env_config": {
        #         "break_after": 40000,
        #     }
        # }
    }

    # Update specific configs for temporary testing
    if hyperopt:
        # tune_config["env_config"]["loglevel"] = 1 

        tune_config["num_workers"] = tune.choice([1, 2, 5, 10])
        tune_config["lr"] = tune.grid_search([0.0001, 0.05])
        tune_config["vf_clip_param"] = tune.choice([1, 10, 1000, 1000])
        tune_config["entropy_coeff"] = tune.choice([0, 0.01, 0.1, 1])
        tune_config["kl_target"] = tune.choice([0.001, 0.01, 0.1])
        tune_config["clip_param"] = tune.choice([0.01, 0.1, 0.5, 1.0])
        # tune_config["model"]["custom_model_config"]["rack_inject"] = tune.choice([True, False])
        # tune_config["model"]["custom_model_config"]["train_batch_size"] = tune.choice([200, 1000, 5000])
        # tune_config["model"]["custom_model_config"]["n_hidden"] = tune.choice([16, 64, 512])
        # tune_config["model"]["custom_model_config"]["conv_filter_size"] = tune.choice([3, 7, 11, 15])
        # tune_config["model"]["custom_model_config"]["n_conv_layers"] = tune.choice([1, 2, 3])
        # tune_config["model"]["custom_model_config"]["activation"] = tune.choice(["relu", "tanh", "elu"])


    analysis = tune.run(
        alg, 
        config=tune_config,

        stop={
            "timesteps_total": timesteps,
        }, 

        # Logging directories
        name=datetime.now().strftime("%y%m%d_%H%M%S_") + tag,
        local_dir=os.path.join(os.path.expanduser("~"), "results", "DCEnv", "RAFSINE" if rafsine else "SIMPLE", alg, model),
        trial_name_creator=lambda trial: "trial",

        # Tuning
        num_samples=n_samples,
        metric="episode_reward_mean",
        mode="max",

        # Checkpointing
        checkpoint_freq=100, # 200 horizon? or is it on training iterations that are 8 * 200 long?
        checkpoint_at_end=True,
        restore = restore,

        # Printing
        verbose=verbose,
    )
    
    return analysis


if __name__ == "__main__":
    # Init ray with all resources
    # needs $ ray start --head --num-cpus=32 --num-gpus=10
    ray.init(address="auto")

    args = parse_all()
    analysis = run(**vars(args))

# best_trial = analysis.best_trial  # Get best trial
# best_config = analysis.best_config  # Get best trial's hyperparameters
# best_logdir = analysis.best_logdir  # Get best trial's logdir
# best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
# best_result = analysis.best_result  # Get best trial's last results
# best_result_df = analysis.best_result_df  # Get best result as pandas dataframe
# 
# dfs = analysis.trial_dataframes

# print(best_trial)
# print(best_config)
# print(best_logdir)
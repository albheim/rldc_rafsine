import os
import argparse
from datetime import datetime

import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog

from loads.temperatures import SinusTemperature
from loads.workloads import ConstantArrival
from util.loggingcallbacks import LoggingCallbacks
from dc.dc import DCEnv
from models.serverconv import ServerConvNetwork

import models

parser = argparse.ArgumentParser()
# Agent settings
parser.add_argument("--model", type=str, default="")
parser.add_argument("--crah_out_setpoint", type=float, default=22)
parser.add_argument("--crah_flow_setpoint", type=float, default=0.8)

# Env settings
parser.add_argument("--rafsine", action="store_true")
parser.add_argument("--seed", type=int, default=37)
parser.add_argument("--avg_load", type=float, default=200)
parser.add_argument("--load_size", type=float, default=20)
parser.add_argument("--actions", nargs="+", default=["server", "crah_out", "crah_flow"])
parser.add_argument("--observations", nargs="+", default=["temp_out", "load", "job"])
parser.add_argument("--ambient", nargs=2, type=float, default=[20, 0])
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--horizon", type=int, default=200)

# Training settings
parser.add_argument("--tag", type=str, default="default")
parser.add_argument("--n_envs", type=int, default=1) # envs for each ppo agent
parser.add_argument("--pretrain_timesteps", type=int, default=0)
parser.add_argument("--stop_timesteps", type=int, default=500000)

args = parser.parse_args()

n_servers = 360
n_racks = 12
n_crah = 4

# Job load
dt = 1
duration = dt * args.avg_load * n_servers / args.load_size
def load_generator_creator():
    return ConstantArrival(load=args.load_size, duration=duration)

# Ambient temp
def temp_generator_creator():
    return SinusTemperature(offset=args.ambient[0], amplitude=args.ambient[1])

# Init ray with all resources
# needs $ ray start --head --num-cpus=20 --num-gpus=1
ray.init(address="auto")

# Register env with ray
tune.register_env("DCEnv", DCEnv)

# Register model with ray
ModelCatalog.register_custom_model("serverconv", ServerConvNetwork)

analysis = tune.run(
    "PPO", 
    config={
        # Environment
        "env": "DCEnv",
        "env_config": {
            "dt": dt,
            "rafsine_flow": args.rafsine,
            "seed": args.seed,
            "n_servers": n_servers,
            "n_racks": n_racks,
            "n_crah": n_crah,
            "n_place": n_servers,
            "load_generator": load_generator_creator,
            "ambient_temp": temp_generator_creator,
            "actions": args.actions,
            "observations": args.observations,
            "pretrain_timesteps": args.pretrain_timesteps,
            "crah_out_setpoint": args.crah_out_setpoint,
            "crah_flow_setpoint": args.crah_flow_setpoint,
        },

        # Model
        "model": {
            "custom_model": args.model,
            "custom_model_config": {
                "n_servers": n_servers,
                "activation": tune.choice(["relu", "tanh"]),
                "n_hidden": tune.choice([32, 128, 512]),
                "n_pre_layers": tune.choice([0, 1, 3]), 
                "inject": tune.choice([True, False]), # If true, pre is injected into server conv
                "n_conv_layers": tune.choice([0, 1, 3]),
                "n_conv_hidden": tune.choice([1, 2, 4]),
                "n_crah_layers": tune.choice([0, 1, 3]),
                "n_value_layers": tune.choice([0, 1, 3]),
            },
        },

        # Worker setup
        "num_workers": args.n_envs, # How many workers are spawned, ppo use this many envs and data is aggregated from all
        "num_envs_per_worker": 1, # How many envs on each worker?
        "num_gpus_per_worker": 1 if args.rafsine else 0, # Only give gpu to rafsine
        "num_cpus_per_worker": 1, # Does this make any difference?
        "seed": args.seed,

        # For logging (does soft_horizon do more, not sure...)
        "callbacks": LoggingCallbacks,
        "soft_horizon": True,
        "no_done_at_end": True,
        "horizon": args.horizon, # Decides length of episodes, should for now be same as rollout_frament_length
        "train_batch_size": args.horizon * args.n_envs, # Collects batch of data from different workes, does min/max/avg over it and trains on it
        "rollout_fragment_length": args.horizon, # How much data is colelcted by each worker before sending in data for training

        # Agent settings
        "vf_clip_param": tune.choice([100.0, 10000.0, 1000000.0]), # Set this to be around the size of value function? Git issue about this not being good, just set high?


        # Data settings
        #"observation_filter": "MeanStdFilter", # Test this
        #"normalize_actions": True,
        #"checkpoint_at_end": True,
    },
    stop={
        "timesteps_total": args.stop_timesteps,
    }, 
    name=args.tag + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"),
    local_dir=os.path.join("results", "PPO", "RAFSINE" if args.rafsine else "SIMPLE", "DCEnv"),
    trial_name_creator=lambda trial: "trial",
    num_samples=args.n_samples,
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max",
    verbose=1,
)

best_trial = analysis.best_trial  # Get best trial
best_config = analysis.best_config  # Get best trial's hyperparameters
best_logdir = analysis.best_logdir  # Get best trial's logdir
best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
best_result = analysis.best_result  # Get best trial's last results
best_result_df = analysis.best_result_df  # Get best result as pandas dataframe

print(best_trial)
print(best_config)
print(best_logdir)
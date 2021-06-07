import os
import argparse
from datetime import datetime

import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog

from loads.temperatures import SinusTemperature
from loads.workloads import RandomArrival
from util.loggingcallbacks import LoggingCallbacks
from dc.dc import DCEnv
from models.serverconv import ServerConvNetwork
from models.emptynet import EmptyNetwork

parser = argparse.ArgumentParser()

# Env settings
parser.add_argument("--rafsine", action="store_true", help="If flag is set the rafsine backend will be used, otherwise the simple simulation is used.")
parser.add_argument("--crah_out_setpoint", type=float, default=22)
parser.add_argument("--crah_flow_setpoint", type=float, default=0.8)

parser.add_argument("--outdoor_temp", nargs=2, type=float, default=[18, 5], help="x[0] + x[1]*sin(t/day) ourdoors temperature")
parser.add_argument("--avg_load", type=float, default=200)
parser.add_argument("--load_size", type=float, default=20)
parser.add_argument("--job_p", type=float, default=0.5, help="Probability that a job arrives each time instance.")

# Training settings
parser.add_argument("--model", type=str, default="serverconv")
parser.add_argument("--seed", type=int, default=-1, help="Seed used for everything, should make the simulations completely reproducible.")
parser.add_argument("--tag", type=str, default="default")
parser.add_argument("--n_envs", type=int, default=1) # envs for each ppo agent
parser.add_argument("--timesteps", type=int, default=500000)
#parser.add_argument("--resume", type=str, default="", help="String with path to run to resume.")
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--horizon", type=int, default=200)

args = parser.parse_args()

n_servers = 360
n_racks = 12
n_crah = 4

# Job load
dt = 1
duration = dt * args.avg_load * n_servers / (args.load_size * args.job_p)
def load_generator_creator():
    return RandomArrival(load=args.load_size, duration=duration, p=args.job_p)

# Outdoor temp
def temp_generator_creator():
    return SinusTemperature(offset=args.outdoor_temp[0], amplitude=args.outdoor_temp[1])

# Init ray with all resources
# needs $ ray start --head --num-cpus=20 --num-gpus=1
ray.init(address="auto")

# Register env with ray
tune.register_env("DCEnv", DCEnv)

# Register model with ray
ModelCatalog.register_custom_model("serverconv", ServerConvNetwork)
ModelCatalog.register_custom_model("baseline", EmptyNetwork)

seed = args.seed if args.seed != -1 else tune.choice([i for i in range(100)])

analysis = tune.run(
    "PPO", 
    config={
        # Environment
        "env": "DCEnv",
        "env_config": {
            "dt": dt,
            "rafsine_flow": args.rafsine,
            "seed": seed,
            "n_servers": n_servers,
            "n_racks": n_racks,
            "n_crah": n_crah,
            "baseline": args.model == "baseline",
            "load_generator": load_generator_creator,
            "outdoor_temp": temp_generator_creator,
            "crah_out_setpoint": args.crah_out_setpoint,
            "crah_flow_setpoint": args.crah_flow_setpoint,
        },

        # Model
        "model": {
            "custom_model": args.model,
            "custom_model_config": {
                "n_servers": n_servers,
                "activation": "tanh", #tune.choice(["relu", "tanh"]),
                "n_hidden": 64, #tune.choice([32, 128, 512]),
                "n_pre_layers": 0, #tune.choice([0, 1, 3]), 
                "inject": False, #tune.choice([True, False]), # If true, pre is injected into server conv
                "n_conv_layers": 1, #tune.choice([0, 1, 3]),
                "n_conv_hidden": 3, #tune.choice([1, 3]),
                "n_crah_layers": 1, #tune.choice([0, 1, 3]),
                "n_value_layers": 2, #tune.choice([0, 1]),
            },
        },

        # Worker setup
        "num_workers": args.n_envs, # How many workers are spawned, ppo use this many envs and data is aggregated from all
        "num_envs_per_worker": 1, # How many envs on each worker?
        "num_gpus_per_worker": 1 / args.n_envs if args.rafsine else 0, # Only give gpu to rafsine
        "num_cpus_per_worker": 1, # Does this make any difference?
        "seed": seed,

        # For logging (does soft_horizon do more, not sure...)
        "callbacks": LoggingCallbacks,
        "soft_horizon": True,
        "no_done_at_end": True,
        "horizon": args.horizon, # Decides length of episodes, should for now be same as rollout_frament_length

        # Training
        "train_batch_size": args.horizon * args.n_envs, # Collects batch of data from different workes, does min/max/avg over it and trains on it
        "rollout_fragment_length": args.horizon, # How much data is colelcted by each worker before sending in data for training

        # Agent settings
        "vf_clip_param": 1000.0, # Set this to be around the size of value function? Git issue about this not being good, just set high?


        # Data settings
        #"observation_filter": "MeanStdFilter", # Test this
        #"normalize_actions": True,
        #"checkpoint_at_end": True,
    },
    stop={
        "timesteps_total": args.timesteps,
    }, 

    # Logging directories
    name=args.tag + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"),
    local_dir=os.path.join("results", "DCEnv", "RAFSINE" if args.rafsine else "SIMPLE", "PPO"),
    trial_name_creator=lambda trial: "trial",

    # Tuning
    num_samples=args.n_samples,
    metric="episode_reward_mean",
    mode="max",

    checkpoint_at_end=True,
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
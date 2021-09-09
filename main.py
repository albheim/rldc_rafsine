import os
import argparse
from datetime import datetime
import numpy as np

import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog

from util.loggingcallbacks import LoggingCallbacks
from dc.dc import DCEnv
from dc.dc_continuous import DCEnvContinuous
from models.serverconv import ServerConvNetwork
from models.serverconv2d import ServerConv2DNetwork
from models.emptynet import EmptyNetwork
from trainers.trainer_config import trainer_config

parser = argparse.ArgumentParser()

# Env settings
parser.add_argument("--rafsine", action="store_true", help="If flag is set the rafsine backend will be used, otherwise the simple simulation is used.")
parser.add_argument("--crah_out_setpoint", type=float, default=22)
parser.add_argument("--crah_flow_setpoint", type=float, default=0.8)
parser.add_argument("--n_bins", type=int, default=0)

parser.add_argument("--outdoor_temp", nargs=2, type=float, default=[18, 5], help="x[0] + x[1]*sin(t/day) ourdoors temperature")
parser.add_argument("--avg_load", type=float, default=200)
parser.add_argument("--job_p", type=float, default=0.5, help="Probability that a job arrives each time instance.")

# Training settings
parser.add_argument("--model", type=str, default="serverconv2d")
parser.add_argument("--alg", type=str, default="PPO")
parser.add_argument("--seed", type=int, default=-1, help="Seed used for everything, should make the simulations completely reproducible.")
parser.add_argument("--tag", type=str, default="default")
parser.add_argument("--n_envs", type=int, default=1) # envs for each ppo agent
parser.add_argument("--timesteps", type=int, default=500000)
#parser.add_argument("--resume", type=str, default="", help="String with path to run to resume.")
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--output", type=str, default=None)

# Other settings
parser.add_argument("--log_full", action="store_true", help="Log all stats for servers, takes much space so use with care.")


args = parser.parse_args()

# Init ray with all resources
# needs $ ray start --head --num-cpus=20 --num-gpus=1
ray.init(address="auto")

# Register env with ray
tune.register_env("DCEnv", DCEnv)
tune.register_env("DCEnvContinuous", DCEnvContinuous)

# Register model with ray
ModelCatalog.register_custom_model("serverconv", ServerConvNetwork)
ModelCatalog.register_custom_model("serverconv2d", ServerConv2DNetwork)
ModelCatalog.register_custom_model("baseline", EmptyNetwork)

# Have grid search here?
seed = 37 #args.seed if args.seed != -1 else tune.choice([i for i in range(100)])

# Some common config
tune_config = {
    # Environment
    "env": "DCEnv",
    "env_config": {
        "rafsine_flow": args.rafsine,
        "seed": seed,
        "baseline": args.model == "baseline",
        "outdoor_temp": args.outdoor_temp,
        "crah_out_setpoint": args.crah_out_setpoint,
        "crah_flow_setpoint": args.crah_flow_setpoint,
        "avg_load": args.avg_load,
        "log_full": args.log_full,
        "n_bins": args.n_bins,
    },

    # Worker setup
    "num_workers": args.n_envs, # How many workers are spawned, ppo use this many envs and data is aggregated from all
    "num_envs_per_worker": 1, # How many envs on each worker? Can be used to vectorize, probably same as num_workers?
    "num_gpus_per_worker": 1 if args.rafsine else 0, # Only give gpu to rafsine
    "num_cpus_per_worker": 1, # Does this make any difference?
    "seed": seed,

    # For logging (does soft_horizon do more, not sure...)
    "callbacks": LoggingCallbacks,
    "soft_horizon": True,
    "no_done_at_end": True,
    "horizon": args.horizon, # Decides length of episodes, should for now be same as rollout_frament_length
    "output": args.output,

    # Training
    "train_batch_size": 4000, # Collects batch of data from different workes, does min/max/avg over it and trains on it
    "rollout_fragment_length": args.horizon, # How much data is colelcted by each worker before sending in data for training

    # Data settings
    #"observation_filter": "MeanStdFilter", # Test this
    #"normalize_actions": True,
    #"checkpoint_at_end": True,
}
# Trainer specific configs
trainer_config = trainer_config(args, tune_config)
tune_config.update(trainer_config)

# Update specific configs for temporary testing
# tune_config["lr"] = tune.grid_search([0.0001, 0.05])

analysis = tune.run(
    args.alg, 
    config=tune_config,
    stop={
        "timesteps_total": args.timesteps,
    }, 

    # Logging directories
    name=datetime.now().strftime("%y%m%d_%H%M%S_") + args.tag,
    local_dir=os.path.join(os.path.expanduser("~"), "results", "DCEnv", "RAFSINE" if args.rafsine else "SIMPLE", args.alg, args.model),
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

# print(best_trial)
# print(best_config)
# print(best_logdir)
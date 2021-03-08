import argparse

import ray
import ray.tune as tune

import job 
from dc.dc import DCEnv
from basiclogger import LoggingCallbacks

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as_test", action="store_true")
parser.add_argument("--stop_reward", type=float, default=0.0)
parser.add_argument("--stop_iters", type=int, default=100)
parser.add_argument("--stop_timesteps", type=int, default=500000)
parser.add_argument("--load_balanced", action="store_true")
parser.add_argument("--rafsine", action="store_true")
parser.add_argument("--n_servers", type=int, default=360)
parser.add_argument("--avg_load", type=int, default=200)
parser.add_argument("--n_crah", type=int, default=4)
parser.add_argument("--tag", type=str, default="")

args = parser.parse_args()

tag = "testing"
def trial_name_string(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    name = str(trial)
    if args.rafsine:
        name += "_rafsine" 
    else: 
        name += "_simple"
    if args.load_balanced:
        name += "_balanced"
    if args.tag != "":
        name += "_" + args.tag
    return name

# avg_load = load * duration / servers => load = avg_load * servers / duration
duration = 3600
load = args.avg_load * args.n_servers / duration
load_generator = job.ConstantArrival(load=load, duration=duration)

# Init ray with all resources
# needs $ ray start --head --port 6379
ray.init(address="auto")

# Set which env to use
ray.tune.register_env("DCEnv", DCEnv)

config = {
    # Environment
    "env": "DCEnv",
    "env_config": {
        "dt": 1,
        "load_balanced": args.load_balanced,
        "rafsine_flow": args.rafsine,
        "n_servers": args.n_servers,
        "n_crah": args.n_crah,
        "load_generator": load_generator,
    },

    # Worker setup
    "num_workers": 1,
    "num_gpus_per_worker": 1 if args.rafsine else 0, # Only give gpu to rafsine
    "num_cpus_per_worker": 4, # Does this make any difference?

    # For logging (does soft_horizon do more, not sure...)
    "callbacks": LoggingCallbacks,
    "soft_horizon": True,
    "no_done_at_end": True,
    "horizon": 1000, # This sets how often stuff is sampled for the avg/min/max logging
    "train_batch_size": 1000, # This sets how often stuff is logged

    # Agent settings
    "vf_clip_param": 10.0,

    # Data settings
    #"observation_filter": "MeanStdFilter", # Test this
    #"normalize_actions": True,
    #"checkpoint_at_end": True,
}

stop = {
    #"training_iteration": args.stop_iters,
    #"episode_reward_mean": args.stop_reward,
    "timesteps_total": args.stop_timesteps,
}

callbacks = [
]

results = tune.run(
    args.run, 
    config=config, 
    callbacks=callbacks, 
    stop=stop, 
    trial_name_creator=trial_name_string,
    verbose=1)

ray.shutdown()

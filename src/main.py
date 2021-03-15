import argparse

import ray
import ray.tune as tune

import job 
from dc.dc import DCEnv
from loggerutils.loggingcallbacks import LoggingCallbacks

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as_test", action="store_true")
parser.add_argument("--stop_reward", type=float, default=0.0)
parser.add_argument("--stop_iters", type=int, default=100)
parser.add_argument("--stop_timesteps", type=int, default=500000)
parser.add_argument("--rafsine", action="store_true")
parser.add_argument("--avg_load", type=int, default=200)
parser.add_argument("--n_servers", type=int, default=360)
parser.add_argument("--n_crah", type=int, default=4)
parser.add_argument("--n_place", type=int, default=360)
parser.add_argument("--load_variance_cost", type=float, default=0.0)
parser.add_argument("--actions", nargs="+", default=["server", "crah_out", "crah_flow"])
parser.add_argument("--observations", nargs="+", default=["temp_out", "load", "job"])
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--pretrain_timesteps", type=int, default=0)

args = parser.parse_args()

def trial_name_string(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    name = str(trial)
    if args.rafsine:
        name += "_RAFSINE" 
    else: 
        name += "_SIMPLE"
    name += "_ACT_" + "_".join(args.actions)
    name += "_OBS_" + "_".join(args.observations)
    if args.tag != "":
        name += "_TAG_" + args.tag
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
        "rafsine_flow": args.rafsine,
        "n_servers": args.n_servers,
        "n_crah": args.n_crah,
        "n_place": args.n_place,
        "load_generator": load_generator,
        "actions": args.actions,
        "observations": args.observations,
        "load_variance_cost": args.load_variance_cost,
        "pretrain_timesteps": args.pretrain_timesteps,
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
    "vf_clip_param": 10.0, # Set this to be around the size of value function?

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
    verbose=1,
    )

import argparse

import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog

import loads 
from dc.dc import DCEnv
from loggerutils.loggingcallbacks import LoggingCallbacks

from models.fcnet import FullyConnectedNetwork
from models.serverconv import ServerConvNetwork

parser = argparse.ArgumentParser()
# Agent settings
parser.add_argument("--model", type=str, default="")
parser.add_argument("--crah_out_setpoint", type=float, default=22)
parser.add_argument("--crah_flow_setpoint", type=float, default=0.8)

# Env settings
parser.add_argument("--rafsine", action="store_true")
parser.add_argument("--seed", type=int, default=37)
parser.add_argument("--avg_load", type=int, default=200)
parser.add_argument("--n_servers", type=int, default=40)#360)
parser.add_argument("--n_racks", type=int, default=1)#12)
parser.add_argument("--n_crah", type=int, default=1)#4)
parser.add_argument("--n_place", type=int, default=360)
parser.add_argument("--actions", nargs="+", default=["server", "crah_out", "crah_flow"])
parser.add_argument("--observations", nargs="+", default=["temp_out", "load", "job"])

# Training settings
parser.add_argument("--worker_seed", type=int, default=None) # Should make training completely reproducible, but might not work well with multiple workers in PPO
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--n_workers", type=int, default=1)
parser.add_argument("--pretrain_timesteps", type=int, default=0)
parser.add_argument("--stop_timesteps", type=int, default=500000)

args = parser.parse_args()

if args.rafsine:
    vars(args)["n_servers"] = 360
    vars(args)["n_racks"] = 12
    vars(args)["n_crah"] = 4

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

# Job load
# avg_load = load_per_step / step_len * duration / servers => duration = step_len * avg_load * servers / load_per_step
dt = 1
load_per_step = 20
duration = dt * args.avg_load * args.n_servers / load_per_step
load_generator = loads.ConstantArrival(load=load_per_step, duration=duration)

# Ambient temp
temp_generator = loads.ConstantTemperature(temp=15)

# Init ray with all resources
# needs $ ray start --head --port 6379
ray.init(address="auto")

# Register env with ray
ray.tune.register_env("DCEnv", DCEnv)

# Register models with ray
ModelCatalog.register_custom_model("fc", FullyConnectedNetwork)
ModelCatalog.register_custom_model("serverconv", ServerConvNetwork)

config = {
    # Environment
    "env": "DCEnv",
    "env_config": {
        "dt": dt,
        "rafsine_flow": args.rafsine,
        "seed": args.seed,
        "n_servers": args.n_servers,
        "n_racks": args.n_racks,
        "n_crah": args.n_crah,
        "n_place": args.n_place,
        "load_generator": load_generator,
        "ambient_temp": temp_generator,
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
            "n_servers": args.n_servers,
        },
    },

    # Worker setup
    "num_workers": args.n_workers,
    "num_gpus_per_worker": 1 if args.rafsine else 0, # Only give gpu to rafsine
    "num_cpus_per_worker": 1, # Does this make any difference?
    "seed": args.worker_seed,

    # For logging (does soft_horizon do more, not sure...)
    "callbacks": LoggingCallbacks,
    "soft_horizon": True,
    "no_done_at_end": True,
    "horizon": 100, # This sets how often stuff is sampled for the avg/min/max logging, no...
    "train_batch_size": 200 * args.n_workers, # This affects how often stuff is logged, maybe???
    "rollout_fragment_length": 200,

    # Agent settings
    "vf_clip_param": 1000000.0, # Set this to be around the size of value function? Git issue about this not being good, just set high?

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
    "PPO", 
    config=config, 
    callbacks=callbacks, 
    stop=stop, 
    trial_name_creator=trial_name_string,
    verbose=1,
    )

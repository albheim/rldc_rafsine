import argparse

import ray
import ray.tune as tune

from dc import DCEnv
from basiclogger import LoggingCallbacks

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as_test", action="store_true")
parser.add_argument("--stop_reward", type=float, default=0.0)
parser.add_argument("--stop_iters", type=int, default=100)
parser.add_argument("--stop_timesteps", type=int, default=500000)
parser.add_argument("--num_cpus", type=int, default=0)
parser.add_argument("--load_balanced", action="store_true")
parser.add_argument("--rafsine", action="store_true")

args = parser.parse_args()

# Init ray with all resources
ray.init()

# Set which env to use
ray.tune.register_env("DCEnv", DCEnv)

config = {
    # Environment
    "env": "DCEnv",
    "env_config": {
        "dt": 1,
        "load_balanced": not args.load_balanced,
        "rafsine_flow": args.rafsine,
    },

    # Worker setup
    "num_workers": 1,
    "num_gpus_per_worker": 1,
    "num_cpus_per_worker": 4, # Does this make any difference?

    # For logging (does soft_horizon do more, not sure...)
    "callbacks": LoggingCallbacks,
    "soft_horizon": True,
    "no_done_at_end": True,
    "horizon": 100, # This sets how often stuff is sampled for the avg/min/max logging
    "train_batch_size": 1000, # This sets how often stuff is logged

    # Agent settings
    "vf_clip_param": 10.0,

    # Data settings
    #"observation_filter": "MeanStdFilter", # Test this
}

stop = {
    #"training_iteration": args.stop_iters,
    #"episode_reward_mean": args.stop_reward,
    "timesteps_total": args.stop_timesteps,
}

callbacks = [
]

results = tune.run(args.run, config=config, callbacks=callbacks, stop=stop, verbose=1)

ray.shutdown()

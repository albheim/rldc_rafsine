import argparse

import ray
import ray.tune as tune

from dc import DCEnv
from basiclogger import LoggingCallbacks

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-reward", type=float, default=0.0)
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--num-cpus", type=int, default=0)

args = parser.parse_args()

ray.init()

ray.tune.register_env("DCEnv", DCEnv)

config = {
    "env": "DCEnv",
    "callbacks": LoggingCallbacks,
    "num_workers": 1,
    "num_gpus_per_worker": 1,
    "horizon": 100,
    "soft_horizon": True,
    "vf_clip_param": 10.0,
    #"observation_filter": "MeanStdFilter", # Test this
    "env_config": {
        "dt": 1
    },
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

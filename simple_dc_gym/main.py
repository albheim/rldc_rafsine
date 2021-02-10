import argparse

import ray
import ray.tune as tune

from dc import SimpleDCEnv
#from tblogger import TBStateLoggerCallback
from basiclogger import MyCallbacks

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-reward", type=float, default=0.0)
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=500000)
parser.add_argument("--num-cpus", type=int, default=0)

args = parser.parse_args()

ray.init()

ray.tune.register_env("SimpleDCEnv", SimpleDCEnv)

config = {
    "env": "SimpleDCEnv",
    "callbacks": MyCallbacks,
    "num_workers": 1,
    "horizon": 1000,
    "vf_clip_param": 1000.0,
    "soft_horizon": True,
    "env_config": {
        "n_servers": 20,
    },
}

stop = {
    #"training_iteration": args.stop_iters,
    #"episode_reward_mean": args.stop_reward,
    "timesteps_total": args.stop_timesteps,
}

callbacks = [
    # MyCallbacks(),
]

results = tune.run(args.run, config=config, callbacks=callbacks, stop=stop, verbose=1)
trials = results.trials
custom_metrics = trials[0].last_result["custom_metrics"]

ray.shutdown()
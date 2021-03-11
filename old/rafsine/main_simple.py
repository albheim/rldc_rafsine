import argparse

import ray
import ray.tune as tune

from dc_simple import SimpleDCEnv
from balancedwrapper import DCLoadBalancedWrapper
#from tblogger import TBStateLoggerCallback
from basiclogger import LoggingCallbacks

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as_test", action="store_true")
parser.add_argument("--stop_reward", type=float, default=0.0)
parser.add_argument("--stop_iters", type=int, default=100)
parser.add_argument("--stop_timesteps", type=int, default=500000)
parser.add_argument("--num_cpus", type=int, default=0)
parser.add_argument("--load_balanced", action="store_true")

args = parser.parse_args()

ray.init()

ray.tune.register_env("SimpleDCEnv", SimpleDCEnv)

config = {
    "env": "SimpleDCEnv",
    "callbacks": LoggingCallbacks,
    "num_workers": 1,
    "horizon": 1000,
    "vf_clip_param": 1000.0,
    "soft_horizon": True,
    #"normalize_actions": True,
    #"observation_filter": "MeanStdFilter", 
    "env_config": {
        "dt": 1.0,
        "n_servers": 360,
        "load_balanced": not args.load_balanced,
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
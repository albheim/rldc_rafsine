import gym
from gym.spaces import Discrete, Box
import ray
from ray import tune
import numpy as np
import datetime
import tensorflow as tf

from ray.tune.logger import TBXLogger

from dc import SimpleDCEnv

class SimpleDCEnvGymWrapper(gym.Env):
    def __init__(self, config):
        self.dcenv = SimpleDCEnv(config["n_servers"])
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(config["n_servers"]), 
            gym.spaces.Box(-1.0, 1.0, shape=(2,))))
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(3 * config["n_servers"],))
        #print(self.action_space, self.observation_space)
        self.alow = np.array([self.dcenv.crah_min_temp, self.dcenv.crah_min_flow])
        self.ahigh = np.array([self.dcenv.crah_max_temp, self.dcenv.crah_max_flow])
        self.slow = np.concatenate(
            (self.dcenv.ambient_temp * np.ones(config["n_servers"]),
            self.dcenv.server_idle_flow * np.ones(config["n_servers"]),
            self.dcenv.server_idle_load * np.ones(config["n_servers"])))
        self.shigh = np.concatenate(
            (self.dcenv.server_max_temp_cpu * np.ones(config["n_servers"]),
            self.dcenv.server_max_flow * np.ones(config["n_servers"]),
            self.dcenv.server_max_load * np.ones(config["n_servers"])))
    
    def act_transform(self, action):
        a1, a2 = action
        a1 = np.random.randint(self.dcenv.n_servers)
        a2 = (a2 * (self.ahigh - self.alow) + self.ahigh + self.alow) / 2
        self.a2 = a2
        return (a1, a2)
    
    def state_transform(self, state):
        state = np.concatenate((state["out_temp"], state["flow"], state["load"]))
        state = (2 * state - self.slow - self.shigh) / (self.shigh - self.slow)
        return state
    
    def reset(self):
        state = self.dcenv.reset()
        return self.state_transform(state)

    def step(self, action):
        action = self.act_transform(action)
        state, reward = self.dcenv.step(action)

        return self.state_transform(state), reward, False, {}

    #def render(self, mode="debug"):
        #print("state: {}  reward: {}  action: {}".format(state, reward, info))

if __name__ == "__main__":
    ray.init(num_cpus=4)
    tune.run(
        "PPO",
        config={
            "env": SimpleDCEnvGymWrapper,
            "num_workers": 3,
            #"normalize_actions": True,
            #"clip_actions": False,
            "horizon": 1000,
            "soft_horizon": True,
            "env_config": {
                "n_servers": 20
            },
            "vf_clip_param": tune.uniform(0.1, 10),
            # "clip_param": 0.3,
            # "entropy_coeff": 0.0,
            # "lambda": 1.0,
            # "kl_coeff": 0.2,
        }
    )
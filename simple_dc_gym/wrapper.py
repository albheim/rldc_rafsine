import gym
from gym.spaces import Discrete, Box
from ray import tune
import numpy as np

from dc import SimpleDCEnv

class SimpleDCEnvGymWrapper(gym.Env):
    def __init__(self, config):
        self.dcenv = SimpleDCEnv(config["n_servers"])
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(config["n_servers"]), 
            gym.spaces.Box(np.array([self.dcenv.crah_min_temp, self.dcenv.crah_min_flow]), np.array([self.dcenv.crah_max_temp, self.dcenv.crah_max_flow]))))
        self.observation_space = gym.spaces.Box(np.concatenate(
            (self.dcenv.ambient_temp * np.ones(config["n_servers"]),
            self.dcenv.server_idle_flow * np.ones(config["n_servers"]),
            self.dcenv.server_idle_load * np.ones(config["n_servers"]))), np.concatenate(
            (self.dcenv.server_max_temp_cpu * np.ones(config["n_servers"]),
            self.dcenv.server_max_flow * np.ones(config["n_servers"]),
            self.dcenv.server_max_load * np.ones(config["n_servers"]))))

    def reset(self):
        state = self.dcenv.reset()
        return np.concatenate((state["out_temp"], state["flow"], state["load"]))

    def step(self, action):
        a1, a2 = action
        a2[0] = 23
        action = (a1, a2)
        state, reward = self.dcenv.step(action)
        print(action, reward)
        return np.concatenate((state["out_temp"], state["flow"], state["load"])), reward, False, {}

    #def render(self, mode="debug"):
        #print("state: {}  reward: {}  action: {}".format(state, reward, info))

if __name__ == "__main__":
    tune.run(
        "PPO",
        config={
            "env": SimpleDCEnvGymWrapper,
            "num_workers": 1,
            "horizon": 1000,
            "soft_horizon": True,
            "env_config": {
                "n_servers": 2
            },
            "vf_clip_param": tune.uniform(0.1, 10),
            "clip_param": 0.3,
            "entropy_coeff": 0.0,
            "lambda": 1.0,
            "kl_coeff": 0.2,
        }
    )
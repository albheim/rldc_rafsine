import numpy as np
import gym
from gym.spaces import Discrete, Box

from dc import DCEnv
from job import ConstantArrival

class DCEnvGymWrapper(gym.Env):
    def __init__(self, config):
        self.dc = DCEnv() # Only inits constants, does not start Sim
        self.load = ConstantArrival(load=20, duration=200)
    
        self.n_servers = len(self.dc.servers)
        self.n_sensors = len(self.dc.sensors)
        self.action_space = gym.spaces.Discrete(self.n_servers)
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(self.n_sensors + self.n_servers,))
        
        self.states = ["temps", "loads"]

        self.slow = np.concatenate((20 * np.ones(self.n_sensors), self.dc.idle_load * np.ones(self.n_servers)))
        self.shigh = np.concatenate((80 * np.ones(self.n_sensors), self.dc.max_load * np.ones(self.n_servers)))

    def state_transform(self, state):
        state = np.concatenate(tuple(state[s] for s in self.states))
        state = (2 * state - self.slow - self.shigh) / (self.shigh - self.slow)
        return state
    
    def reset(self):
        state = self.dc.reset()
        return self.state_transform(state)

    def step(self, action):
        action = (action, self.load.step())
        state, reward = self.dc.step(action)
        print(reward)
        return self.state_transform(state), reward, False, {}

if __name__ == "__main__":
    import ray
    from ray.rllib.agents import ppo
    ray.init()
    trainer = ppo.PPOTrainer(
        env = DCEnvGymWrapper, 
        config = {
            "num_workers": 1,
            "num_gpus": 1, 
            "env_config": {}
        }
    )
    for i in range(100):
        print(trainer.train())

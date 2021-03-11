import gym
import numpy as np

class DCLoadBalancedWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DCLoadBalancedWrapper, self).__init__(env)

    def action(self, action):
        placement = np.argmin(self.env.server_load)
        return placement, action
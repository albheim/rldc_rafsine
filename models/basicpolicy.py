# https://docs.ray.io/en/master/rllib-concepts.html
from ray.rllib.policy import Policy
import numpy as np

class RandomPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        return {}  

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

class DefaultPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.crah_out = 0.0
        self.crah_flow = 0.8

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        print(obs_batch)
        return [(np.argmin(obs_batch[0]), np.array([self.crah_out, self.crah_flow])) for obs in obs_batch], [], {}

    def learn_on_batch(self, samples):
        return {}  

    def get_weights(self):
        return {"co": self.crah_out, "cf": self.crah_flow}

    def set_weights(self, weights):
        pass

from ray.rllib.agents.trainer_template import build_trainer

# <class 'ray.rllib.agents.trainer_template.MyCustomTrainer'>
DefaultTrainer = build_trainer(
    name="DefaultTrainer",
    default_policy=DefaultPolicy)
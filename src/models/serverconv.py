import numpy as np
import gym
import tensorflow as tf

import ray
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_ops import one_hot

class ServerConvNetwork(TFModelV2):
    # https://docs.ray.io/en/master/rllib-models.html#more-examples-for-building-custom-models

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.original_space = obs_space.original_space if \
            hasattr(obs_space, "original_space") else obs_space
        assert isinstance(self.original_space, (gym.spaces.Tuple)), \
            "`obs_space.original_space` must be Tuple!"

        super().__init__(self.original_space, action_space, num_outputs,
                         model_config, name)

        self.n_servers = model_config["custom_model_config"]["n_servers"]
        # Build the CNN(s) given obs_space's image components.
        inputs = []
        server_reshaped = []
        for component in self.original_space:
            # (batch_size, n_servers, n_features)
            inp = tf.keras.layers.Input(shape=component.shape)
            inputs += inp
            if component.shape[0] == self.n_servers:
                server_reshaped += tf.keras.layers.Reshape(shape=(-1, 1))(inp)

        server_conc = tf.keras.layers.Concatenate(axis=2)(server_reshaped)
        prob_conv = tf.keras.layers.Conv1D(1, 1)(server_conc)
        prob_reshaped = tf.keras.layers.Reshape((-1,))(prob_conv)
        
        all_conc = tf.keras.layers.Concatenate()(inputs)
        dense1 = tf.keras.layers.Dense(256, activation='relu')(all_conc)
        dense2 = tf.keras.layers.Dense(256, activation='relu')(dense1)
        dense_out = tf.keras.layers.Dense(2)(dense2)

        action_out = tf.keras.layers.Concatenate()([prob_reshaped, dense_out])

        dense1 = tf.keras.layers.Dense(256, activation='relu')(all_conc)
        dense2 = tf.keras.layers.Dense(256, activation='relu')(dense1)
        dense_out = tf.keras.layers.Dense(1)(dense2)

        self.base_model = tf.keras.Model(inputs=self.inputs, outputs=[action_out, value_out])
        self.register_variables(self.base_model.variables)
        

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        orig_obs = input_dict[SampleBatch.OBS]
        logits, values = self.base_model(orig_obs)
        self._value_out = tf.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out
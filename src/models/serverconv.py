import numpy as np
import gym

import ray
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_ops import one_hot
from ray.rllib.utils import try_import_tf

tf, tf2, _ = try_import_tf()

class ServerConvNetwork(TFModelV2):
    # https://docs.ray.io/en/master/rllib-models.html#more-examples-for-building-custom-models

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, n_servers):
        # self.original_space = obs_space.original_space if \
        #     hasattr(obs_space, "original_space") else obs_space
        # assert isinstance(self.original_space, (gym.spaces.Tuple)), \
        #     "`obs_space.original_space` must be Tuple!"
        self.original_space = obs_space

        super(ServerConvNetwork, self).__init__(
            self.original_space, action_space, num_outputs, model_config, name)

        input_temp = tf.keras.layers.Input(shape=(n_servers,))
        input_load = tf.keras.layers.Input(shape=(n_servers,))
        input_job = tf.keras.layers.Input(shape=(2,))
        inputs = [input_temp, input_load, input_job]

        temp_reshaped = tf.keras.layers.Reshape(target_shape=(-1, 1))(input_temp)
        load_reshaped = tf.keras.layers.Reshape(target_shape=(-1, 1))(input_load)
        server_reshaped = [temp_reshaped, load_reshaped]

        server_conc = tf.keras.layers.Concatenate(axis=2)(server_reshaped)
        prob_conv = tf.keras.layers.Conv1D(1, 1, activation='relu')(server_conc)
        prob_flattened = tf.keras.layers.Flatten()(prob_conv)
        
        all_conc = tf.keras.layers.Concatenate()(inputs)
        dense = tf.keras.layers.Dense(256, activation='relu')(all_conc)
        dense = tf.keras.layers.Dense(256, activation='relu')(dense)
        dense_out = tf.keras.layers.Dense(4)(dense)

        action_out = tf.keras.layers.Concatenate()([prob_flattened, dense_out])

        dense = tf.keras.layers.Dense(256, activation='relu')(all_conc)
        dense = tf.keras.layers.Dense(256, activation='relu')(dense)
        value_out = tf.keras.layers.Dense(1)(dense)

        self.base_model = tf.keras.Model(inputs=inputs, outputs=[action_out, value_out])
        self.register_variables(self.base_model.variables)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        orig_obs = input_dict[SampleBatch.OBS]
        logit_tuple, values = self.base_model(orig_obs)
        self._value_out = tf.reshape(values, [-1])
        return logit_tuple, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out


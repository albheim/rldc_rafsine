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

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
            n_servers, n_hidden=256, inject=False, activation='relu',
            n_conv_layers=1, n_conv_hidden=1):
        super(ServerConvNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        input_temp = tf.keras.layers.Input(shape=(n_servers,))
        input_load = tf.keras.layers.Input(shape=(n_servers,))
        input_job = tf.keras.layers.Input(shape=(2,))
        inputs = [input_temp, input_load, input_job]

        all_conc = tf.keras.layers.Concatenate()(inputs)
        server_inject = tf.keras.layers.Dense(n_servers, activation=activation)(all_conc)

        server_inputs = [input_temp, input_load]
        if inject:
            server_inputs += [server_inject]
        
        server_reshaped = [tf.keras.layers.Reshape(target_shape=(-1, 1))(x) for x in server_inputs]

        convlayer = tf.keras.layers.Concatenate(axis=2, name="server_concat")(server_reshaped)
        for i in range(n_conv_layers - 1):
            convlayer = tf.keras.layers.Conv1D(n_conv_hidden, 1, activation=activation, name="server_conv"+str(i))(convlayer)
        prob_conv = tf.keras.layers.Conv1D(1, 1, activation=activation, name="server_conv_last")(convlayer)
        prob_flattened = tf.keras.layers.Flatten(name="server_out")(prob_conv)
        
        dense = tf.keras.layers.Dense(n_hidden, activation=activation, name="crah_dense1")(all_conc)
        dense = tf.keras.layers.Dense(n_hidden, activation=activation, name="crah_dense2")(dense)
        dense_out = tf.keras.layers.Dense(4, name="crah_out")(dense)

        action_out = tf.keras.layers.Concatenate(name="action_out")([prob_flattened, dense_out])

        dense = tf.keras.layers.Dense(n_hidden, activation=activation, name="value_dense1")(all_conc)
        dense = tf.keras.layers.Dense(n_hidden, activation=activation, name="value_dense2")(dense)
        value_out = tf.keras.layers.Dense(1, name="value_out")(dense)

        self.base_model = tf.keras.Model(inputs=inputs, outputs=[action_out, value_out])
        #self.register_variables(self.base_model.variables)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        orig_obs = input_dict[SampleBatch.OBS]
        logit_tuple, values = self.base_model(orig_obs)
        self._value_out = tf.reshape(values, [-1])
        return logit_tuple, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out


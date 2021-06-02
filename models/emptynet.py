from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf
import numpy as np

tf, tf2, _ = try_import_tf()

class EmptyNetwork(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(EmptyNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        inp = tf.keras.Input(shape=obs_space.shape)
        aoutp = tf.keras.layers.Dense(num_outputs)(inp)
        voutp = tf.keras.layers.Dense(1)(inp)
        self.base_model = tf.keras.Model(inputs=inp, outputs=[aoutp, voutp])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        orig_obs = input_dict[SampleBatch.OBS]
        logit_tuple, values = self.base_model(orig_obs)
        self._value_out = tf.reshape(values, [-1])
        return logit_tuple, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out
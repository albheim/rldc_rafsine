from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf
import numpy as np

tf, tf2, _ = try_import_tf()

class EmptyNetwork(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        obs_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        super(EmptyNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        input_job = tf.keras.layers.Input(shape=(1,))

        aoutp = tf.keras.layers.Dense(num_outputs)(input_job)
        voutp = tf.keras.layers.Dense(1)(input_job)

        self.base_model = tf.keras.Model(inputs=[input_job], outputs=[aoutp, voutp])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"][3]
        logit_tuple, values = self.base_model(obs)
        self._value_out = tf.reshape(values, [-1])
        return logit_tuple, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out
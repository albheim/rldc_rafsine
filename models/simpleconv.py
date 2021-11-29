from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

tf, tf2, _ = try_import_tf()
tf = tf2
class SimpleConvNetwork(TFModelV2):
    # https://docs.ray.io/en/master/rllib-models.html#more-examples-for-building-custom-models

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        obs_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        super(SimpleConvNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        #fixed_input = tf.keras.layers.Input(tensor=tf.constant([1, 2, 3, 4]))

        n_servers = kwargs.get("n_servers")
        placement_block_size = kwargs.get("placement_block_size")
        n_crah = kwargs.get("n_crah")
        n_hidden = kwargs.get("n_hidden")
        activation = kwargs.get("activation")
        n_conv_layers = kwargs.get("n_conv_layers")
        n_conv_hidden = kwargs.get("n_conv_hidden")
        n_crah_layers = kwargs.get("n_crah_layers")
        n_value_layers = kwargs.get("n_value_layers")
        conv_filter_size = kwargs.get("conv_filter_size")

        blocks = n_servers // placement_block_size

        input_temp = tf.keras.layers.Input(shape=(blocks,))
        input_load = tf.keras.layers.Input(shape=(blocks,))
        # add constant tensor for height of each server?
        block_inputs = [input_load, input_temp]

        input_outdoor_temp = tf.keras.layers.Input(shape=(1,))
        input_job = tf.keras.layers.Input(shape=(1,))
        other_inputs = [input_outdoor_temp, input_job]

        all_conc = tf.keras.layers.Concatenate()([input_outdoor_temp] + block_inputs)

        # Server inputs to racks * servers * params
        block_reshaped = [tf.keras.layers.Reshape((-1, 1))(layer) for layer in block_inputs]
        block_concat = tf.keras.layers.Concatenate(name="server_concat", axis=2)(block_reshaped)


        convlayer = block_concat
        for i in range(n_conv_layers):
            convlayer = tf.keras.layers.Conv1D(n_conv_hidden, conv_filter_size, padding="same", activation=activation, name=f"server_conv{i}")(convlayer)
        conv_server_out = tf.keras.layers.Conv1D(1, conv_filter_size, padding="same", activation=activation, name="server_conv_last")(convlayer)
        conv_flattened = tf.keras.layers.Flatten(name="server_out")(conv_server_out)
        placement_out = tf.keras.layers.Multiply()([conv_flattened, input_job]) # Removes gradient if zero job was placed

        # Crah settings
        crah_dense = all_conc
        for i in range(n_crah_layers):
            crah_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"crah_dense{i}")(crah_dense)
        crah_dense_out = tf.keras.layers.Dense(4 * n_crah, name="crah_out")(crah_dense)

        # Full action distribution information
        action_out = tf.keras.layers.Concatenate(name="action_out")([placement_out, crah_dense_out])

        # Value net
        value_dense = all_conc
        for i in range(n_value_layers):
            value_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"value_dense{i}")(value_dense)
        value_out = tf.keras.layers.Dense(1, name="value_out")(value_dense)

        self.base_model = tf.keras.Model(inputs=block_inputs + other_inputs, outputs=[action_out, value_out])
        #self.base_model.summary()
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


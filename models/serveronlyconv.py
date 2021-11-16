from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

tf, tf2, _ = try_import_tf()
tf = tf2
class ServerOnlyConvNetwork(TFModelV2):
    # https://docs.ray.io/en/master/rllib-models.html#more-examples-for-building-custom-models

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        obs_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        super(ServerOnlyConvNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        #fixed_input = tf.keras.layers.Input(tensor=tf.constant([1, 2, 3, 4]))

        n_servers = kwargs.get("n_servers", 360)
        n_crah = kwargs.get("n_crah", 4)
        n_hidden = kwargs.get("n_hidden", 256)
        activation = kwargs.get("activation", 'relu')
        n_conv_layers = kwargs.get("n_conv_layers", 1)
        n_conv_hidden = kwargs.get("n_conv_hidden", 1)
        n_crah_layers = kwargs.get("n_crah_layers", 1)
        n_value_layers = kwargs.get("n_value_layers", 2)
        conv_filter_size = kwargs.get("conv_filter_size", 1)

        input_load = tf.keras.layers.Input(shape=(n_servers,))
        input_temp = tf.keras.layers.Input(shape=(n_servers,))
        # add constant tensor for height of each server?
        server_inputs = [input_load, input_temp]

        input_job = tf.keras.layers.Input(shape=(1,))
        other_inputs = [input_job]

        all_conc = tf.keras.layers.Concatenate()(server_inputs)

        # Server inputs to racks * servers * params
        server_reshaped = [tf.keras.layers.Reshape((-1, 1))(layer) for layer in server_inputs]
        server_concat = tf.keras.layers.Concatenate(name="server_concat", axis=2)(server_reshaped)


        convlayer = server_concat
        for i in range(n_conv_layers):
            convlayer = tf.keras.layers.Conv1D(n_conv_hidden, conv_filter_size, padding="same", activation=activation, name=f"server_conv{i}")(convlayer)
        conv_server_out = tf.keras.layers.Conv1D(1, conv_filter_size, padding="same", activation=activation, name="server_conv_last")(convlayer)
        conv_flattened = tf.keras.layers.Flatten(name="server_out")(conv_server_out)
        placement_out = tf.keras.layers.Multiply()([conv_flattened, input_job]) # Removes gradient if zero job was placed

        # Value net
        value_dense = all_conc
        for i in range(n_value_layers):
            value_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"value_dense{i}")(value_dense)
        value_out = tf.keras.layers.Dense(1, name="value_out")(value_dense)

        self.base_model = tf.keras.Model(inputs=server_inputs + other_inputs, outputs=[placement_out, value_out])
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


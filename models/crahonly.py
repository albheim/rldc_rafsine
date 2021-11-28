from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

tf, tf2, _ = try_import_tf()
tf = tf2
class CRAHOnlyNetwork(TFModelV2):
    # https://docs.ray.io/en/master/rllib-models.html#more-examples-for-building-custom-models

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        obs_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        super(CRAHOnlyNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        #fixed_input = tf.keras.layers.Input(tensor=tf.constant([1, 2, 3, 4]))

        n_servers = kwargs.get("n_servers", 360)
        n_crah = kwargs.get("n_crah", 4)
        n_racks = kwargs.get("n_racks", 12)
        n_hidden = kwargs.get("n_hidden", 256)
        activation = kwargs.get("activation", 'relu')
        n_crah_layers = kwargs.get("n_crah_layers", 1)
        n_value_layers = kwargs.get("n_value_layers", 2)
        crah_input = kwargs.get("crah_input", "all")
        value_input = kwargs.get("value_input", "all")

        input_load = tf.keras.layers.Input(shape=(n_servers,))
        input_temp = tf.keras.layers.Input(shape=(n_servers,))
        # add constant tensor for height of each server?
        server_inputs = [input_load, input_temp]

        input_outdoor_temp = tf.keras.layers.Input(shape=(1,))
        other_inputs = [input_outdoor_temp]

        all_conc = tf.keras.layers.Concatenate()(other_inputs + server_inputs)

        # Server inputs to racks * servers * params
        server_reshaped = [tf.keras.layers.Reshape(target_shape=(n_racks, n_servers // n_racks, 1))(x) for x in server_inputs]
        server_concat = tf.keras.layers.Concatenate(name="server_concat")(server_reshaped)

        # Rack based input
        rack_mean_vals = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=2))(server_concat)
        #conv_rack_out = tf.keras.layers.Conv2D(1, (1, n_servers // n_racks), activation=activation, name="conv_rack_for_crah")(server_concat)
        rack_flattened = tf.keras.layers.Flatten(name="rack_flattened")(rack_mean_vals)

        # Server based inputs
        server_mean_vals = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1))(rack_mean_vals)

        # Other input
        other_input = tf.keras.layers.Concatenate(name="other_concat")([input_outdoor_temp, server_mean_vals, rack_flattened])
        
        # Crah settings
        if crah_input == "all":
            crah_dense = all_conc
        else:
            crah_dense = other_input 
        for i in range(n_crah_layers):
            crah_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"crah_dense{i}")(crah_dense)
        crah_dense_out = tf.keras.layers.Dense(4 * n_crah, name="crah_out")(crah_dense)

        # Full action distribution information
        #action_out = tf.keras.layers.Concatenate(name="action_out")([placement_out, crah_dense_out])
        action_out = crah_dense_out

        # Value net
        if value_input == "all":
            value_dense = all_conc
        else:
            value_dense = other_input
        for i in range(n_value_layers):
            value_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"value_dense{i}")(value_dense)
        value_out = tf.keras.layers.Dense(1, name="value_out")(value_dense)

        self.base_model = tf.keras.Model(inputs=server_inputs + other_inputs, outputs=[action_out, value_out])
        self.base_model.summary()
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


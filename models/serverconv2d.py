from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

tf, tf2, _ = try_import_tf()
tf = tf2
class ServerConv2DNetwork(TFModelV2):
    # https://docs.ray.io/en/master/rllib-models.html#more-examples-for-building-custom-models

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
            n_servers=360, n_crah=4, n_hidden=256, inject=False, activation='relu',
            n_conv_layers=1, n_conv_hidden=1, n_pre_layers=1, n_crah_layers=1, n_value_layers=2):
        obs_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        super(ServerConv2DNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        
        #fixed_input = tf.keras.layers.Input(tensor=tf.constant([1, 2, 3, 4]))

        input_load = tf.keras.layers.Input(shape=(n_servers,))
        input_temp = tf.keras.layers.Input(shape=(n_servers,))
        server_inputs = [input_load, input_temp]

        input_outdoor_temp = tf.keras.layers.Input(shape=(1,))
        input_job = tf.keras.layers.Input(shape=(1,))
        other_inputs = [input_outdoor_temp, input_job]

        all_conc = tf.keras.layers.Concatenate()(other_inputs + server_inputs)

        # Pre net, goes both to conv and crah 
        pre = all_conc
        for i in range(n_pre_layers):
            pre = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"pre_dense{i}")(pre)

        # Conv input
        server_inject = tf.keras.layers.Dense(n_servers, activation=activation)(all_conc)
        if inject:
            server_inputs += [server_inject]

        # Job placement
        filter_size=11
        server_reshaped = [tf.keras.layers.Reshape(target_shape=(n_crah, -1, 1))(x) for x in server_inputs]
        server_concat = tf.keras.layers.Concatenate(name="server_concat")(server_reshaped)
        convlayer = server_concat
        for i in range(n_conv_layers):
            convlayer = tf.keras.layers.Conv2D(n_conv_hidden, (1, filter_size), padding="same", activation=activation, name=f"server_conv{i}")(convlayer)
        conv_server_out = tf.keras.layers.Conv2D(1, (1, filter_size), padding="same", activation=activation, name="server_conv_last")(convlayer)
        conv_flattened = tf.keras.layers.Flatten(name="server_out")(conv_server_out)
        placement_out = tf.keras.layers.Multiply()([conv_flattened, input_job]) # Removes gradient if zero job was placed

        # Rack based input
        conv_rack_out = tf.keras.layers.Conv2D(1, (1, n_servers // n_crah), activation=activation, name="rack_conv")(convlayer)
        rack_flattened = tf.keras.layers.Flatten(name="rack_conv_out")(conv_rack_out)
        
        # Crah settings
        server_total_load = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1, keepdims=True))(input_load)
        server_mean_temp = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True))(input_temp)
        crah_input = tf.keras.layers.Concatenate(name="crah_concat")([input_outdoor_temp, server_total_load, server_mean_temp, rack_flattened])
        crah_dense = crah_input # all_conc
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

        self.base_model = tf.keras.Model(inputs=server_inputs + other_inputs, outputs=[action_out, value_out])
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

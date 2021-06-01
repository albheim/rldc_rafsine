from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

tf, tf2, _ = try_import_tf()

class ServerConvNetwork(TFModelV2):
    # https://docs.ray.io/en/master/rllib-models.html#more-examples-for-building-custom-models

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
            n_servers, n_hidden=256, inject=False, activation='relu',
            n_conv_layers=1, n_conv_hidden=1, n_pre_layers=1, n_crah_layers=1, n_value_layers=2, empty=False):
        super(ServerConvNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        
        input_temp = tf.keras.layers.Input(shape=(n_servers,))
        input_load = tf.keras.layers.Input(shape=(n_servers,))
        input_job = tf.keras.layers.Input(shape=(1,))
        inputs = [input_temp, input_load, input_job]

        all_conc = tf.keras.layers.Concatenate()(inputs)

        if empty:
            action_out = tf.keras.layers.Dense(2)(all_conc)
        else:
            # Pre net, goes both to conv and crah 
            pre = all_conc
            for i in range(n_pre_layers):
                pre = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"pre_dense{i}")(pre)

            # Conv input
            server_inject = tf.keras.layers.Dense(n_servers, activation=activation)(all_conc)
            server_inputs = [input_temp, input_load]
            if inject:
                server_inputs += [server_inject]

            # Job placement
            server_reshaped = [tf.keras.layers.Reshape(target_shape=(-1, 1))(x) for x in server_inputs]
            convlayer = tf.keras.layers.Concatenate(axis=2, name="server_concat")(server_reshaped)
            for i in range(n_conv_layers):
                convlayer = tf.keras.layers.Conv1D(n_conv_hidden, 1, activation=activation, name=f"server_conv{i}")(convlayer)
            prob_conv = tf.keras.layers.Conv1D(1, 1, activation=activation, name="server_conv_last")(convlayer)
            prob_flattened = tf.keras.layers.Flatten(name="server_out")(prob_conv)
            
            # Crah settings
            crah_dense = all_conc
            for i in range(n_crah_layers):
                crah_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"crah_dense{i}")(crah_dense)
            crah_dense_out = tf.keras.layers.Dense(4, name="crah_out")(crah_dense)

            # Full action distribution information
            action_out = tf.keras.layers.Concatenate(name="action_out")([prob_flattened, crah_dense_out])

        # Value net
        value_dense = all_conc
        for i in range(n_value_layers):
            value_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"value_dense{i}")(value_dense)
        value_out = tf.keras.layers.Dense(1, name="value_out")(value_dense)

        self.base_model = tf.keras.Model(inputs=inputs, outputs=[action_out, value_out])
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


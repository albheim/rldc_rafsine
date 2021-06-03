from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

tf, tf2, _ = try_import_tf()
tf = tf2

# Assumes dict spaces
class ServerConvNetwork(TFModelV2):
    # https://docs.ray.io/en/master/rllib-models.html#more-examples-for-building-custom-models

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
            n_servers=360, n_hidden=256, inject=False, activation='relu',
            n_conv_layers=1, n_conv_hidden=1, n_pre_layers=1, n_crah_layers=1, n_value_layers=2):

        obs_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        super(ServerConvNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        input_dict = {}
        all_inputs = []
        for component_name in obs_space:
            inp = tf.keras.Input(shape=obs_space[component_name].shape, name=component_name)
            input_dict[component_name] = inp
            all_inputs.append(inp)

        all_conc = tf.keras.layers.Concatenate()(all_inputs)

        server_total_load = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1, keepdims=True))(input_dict["server_load"])
        server_mean_temp = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True))(input_dict["server_temp_out"])
        server_max_temp = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1, keepdims=True))(input_dict["server_temp_out"])
        other_conc = tf.keras.layers.Concatenate(name="crah_concat")([input_dict["outdoor_temp"], server_total_load, server_mean_temp, server_max_temp])

        # Pre net, goes both to conv and crah 
        pre = all_conc
        for i in range(n_pre_layers):
            pre = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"pre_dense{i}")(pre)

        # Inject full info into conv? Probably not good idea?
        server_inputs = [input_dict[k] for k in input_dict if "server" in k]
        if inject:
            server_inputs.append(tf.keras.layers.Dense(n_servers, activation=activation)(all_conc))

        # Job placement
        server_reshaped = [tf.keras.layers.Reshape(target_shape=(-1, 1))(x) for x in server_inputs]
        convlayer = tf.keras.layers.Concatenate(axis=2, name="server_concat")(server_reshaped)
        for i in range(n_conv_layers):
            convlayer = tf.keras.layers.Conv1D(n_conv_hidden, 1, activation=activation, name=f"server_conv{i}")(convlayer)
        conv_out = tf.keras.layers.Conv1D(1, 1, activation=activation, name="server_conv_last")(convlayer)
        conv_flattened = tf.keras.layers.Flatten(name="server_out")(conv_out)
        placement_out = tf.keras.layers.Multiply()([conv_flattened, input_dict["job"]]) # Removes gradient if zero job was placed
        
        # Crah settings
        crah_dense = other_conc 
        #crah_dense = all_conc
        for i in range(n_crah_layers):
            crah_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"crah_dense{i}")(crah_dense)
        crah_temp_out = tf.keras.layers.Dense(2, name="crah_temp_out")(crah_dense)
        crah_flow_out = tf.keras.layers.Dense(2, name="crah_flow_out")(crah_dense)

        # Select only desired actions to inlucde in output
        fake_out = tf.keras.layers.Dense(num_outputs)(all_inputs[0]) # Not good way
        all_action_outputs = {"crah_temp_out": crah_temp_out, "crah_flow": crah_flow_out, "server_placement": placement_out, "none": fake_out}
        action_outputs = [all_action_outputs[k] for k in action_space]
        action_out = tf.keras.layers.Concatenate(name="action_out")(action_outputs) if len(action_outputs) > 1 else action_outputs[0]

        # Value net
        value_dense = other_conc 
        #value_dense = all_conc
        for i in range(n_value_layers):
            value_dense = tf.keras.layers.Dense(n_hidden, activation=activation, name=f"value_dense{i}")(value_dense)
        value_out = tf.keras.layers.Dense(1, name="value_out")(value_dense)

        self.base_model = tf.keras.Model(inputs=input_dict, outputs=[action_out, value_out])
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


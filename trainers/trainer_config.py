from ray import tune

def trainer_config(args, config):
    n_servers = config["env_config"].get("n_servers", 360)
    if args.alg == "PPO":
        return {
            # Model
            "model": {
                "custom_model": args.model,
                "custom_model_config": {
                    "n_servers": n_servers,
                    "activation": "elu", #tune.choice(["relu", "tanh", "elu"]),
                    "n_hidden": 64, #tune.choice([32, 128, 512]),
                    "n_pre_layers": 0, #tune.choice([0, 1, 3]), 
                    "inject": False, #tune.choice([True, False]), # If true, pre is injected into server conv
                    "rack_inject": "True", #tune.choice([True, False]), # If true, pre is injected into server conv
                    "conv_filter_size": 11, #tune.choice([1, 5, 11]),
                    "filter_size": 5,
                    "n_conv_layers": 1, #tune.choice([0, 1, 3]),
                    "n_conv_hidden": 3, #tune.choice([1, 3]),
                    "n_crah_layers": 1, #tune.choice([0, 1, 3]),
                    "n_value_layers": 2, #tune.choice([0, 1]),
                    "crah_input": "other", #tune.choice(["all", "other"]),
                    "value_input": "all", #tune.choice(["all", "other"]),
                },
            },

            # Worker setup
            "num_workers": args.n_envs, # How many workers are spawned, ppo use this many envs and data is aggregated from all

            # Training
            "train_batch_size": 4000, #tune.choice([4000, 1000, 10000]), # Collects batch of data from different workes, does min/max/avg over it and trains on it
            "rollout_fragment_length": args.horizon, # How much data is colelcted by each worker before sending in data for training

            # Agent settings
            "vf_clip_param": 1000.0, #tune.choice([1.0, 10.0, 100.0, 1000.0]), # Set this to be around the size of value function? Git issue about this not being good, just set high?
            "entropy_coeff": 0, #tune.choice([0, 1]),
            "kl_target": 0.01, #tune.choice([0.01, 0.001, 0.1]),
            "clip_param": 0.3, #tune.choice([0.3, 0.03, 3]),

        }
    elif args.alg == "baseline":
        return {

        }
    else:
        pass

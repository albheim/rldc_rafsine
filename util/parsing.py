import argparse

def parse_all():
    parser = argparse.ArgumentParser()

    # Env settings
    parser.add_argument("--rafsine", action="store_true", help="If flag is set the rafsine backend will be used, otherwise the simple simulation is used.")
    parser.add_argument("--crah_out_setpoint", type=float, default=19)
    parser.add_argument("--crah_flow_setpoint", type=float, default=0.8)
    parser.add_argument("--n_bins", type=int, default=0)
    parser.add_argument("--break_after", type=float, default=-1)

    parser.add_argument("--avg_load", type=float, default=100)
    parser.add_argument("--job_p", type=float, default=0.5, help="Probability that a job arrives each time instance.")

    # Training settings
    parser.add_argument("--model", type=str, default="serverconv2d")
    parser.add_argument("--alg", type=str, default="PPO")
    parser.add_argument("--seed", type=int, default=37, help="Seed used for everything, should make the simulations completely reproducible.")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--n_envs", type=int, default=1) # envs for each ppo agent
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--restore", type=str, default=None, help="String with path to checkpoint dir")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--hyperopt", action="store_true")

    # Other settings
    parser.add_argument("--loglevel", type=int, default=1, help="0 no extra, 1 most stuff, 2 all stuff")

    args = parser.parse_args()
    return args
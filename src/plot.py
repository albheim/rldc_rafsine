import numpy as np
import matplotlib.pyplot as plt

def plot_all(hist):
    n_servers = len(hist[0][0]["load"])

    fig, axs = plt.subplots(4, 2, figsize=(25, 15))

    ts = [h[0]["time"] for h in hist]
    axs[0, 0].plot(ts, [h[0]["in_temp"] for h in hist])
    axs[0, 0].plot(ts, [h[0]["out_temp"] for h in hist])
    axs[0, 0].plot(ts, [h[0]["cpu_temp"] for h in hist])
    axs[0, 0].set_ylabel("temperature [C]")
    axs[0, 0].set_xlabel("time [s]")
    axs[0, 0].legend(["servers in"] +
                     ["server_{:02} out".format(i+1) for i in range(n_servers)] +
                     ["server_{:02} cpu".format(i+1) for i in range(n_servers)])

    axs[0, 1].plot(ts, [h[0]["cost"] for h in hist])
    axs[0, 1].set_ylabel("cost")
    axs[0, 1].set_xlabel("time [s]")

    axs[1, 0].plot(ts, [h[0]["load"] for h in hist])
    axs[1, 0].set_ylabel("load [W]")
    axs[1, 0].set_xlabel("time [s]")
    axs[1, 0].legend(["server_{:02}".format(i+1) for i in range(n_servers)])

    rew = [h[2] for h in hist]
    span = 500
    axs[1, 1].plot(ts, rew)
    axs[1, 1].plot(ts, np.convolve(rew, np.ones(
        span * 2 + 1) / (span * 2 + 1), mode="same"))
    axs[1, 1].set_ylabel("reward")
    axs[1, 1].set_xlabel("time [s]")

    axs[2, 0].plot(ts, [h[0]["flow"] for h in hist])
    axs[2, 0].set_ylabel("flow [m3/s]")
    axs[2, 0].set_xlabel("time [s]")
    axs[2, 0].legend(["server_{:02}".format(i+1) for i in range(n_servers)])

    axs[2, 1].plot(ts, [h[0]["generator"] for h in hist])
    axs[2, 1].set_ylabel("generator power [w]")
    axs[2, 1].set_xlabel("time [s]")

    axs[3, 0].plot(ts, [h[0]["crah_temp_out"] for h in hist])
    axs[3, 0].set_ylabel("crah temp")
    axs[3, 0].set_xlabel("time [s]")

    axs[3, 1].plot(ts, [h[0]["crah_flow"] for h in hist])
    axs[3, 1].set_ylabel("crah flow")
    axs[3, 1].set_xlabel("time [s]")

    plt.show()

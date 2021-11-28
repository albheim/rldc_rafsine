import os
import glob
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def get_tblogs(data_path):
    search_path = os.path.join(os.path.expanduser("~"), "results", data_path, "event*")
    exp_path = glob.glob(search_path)
    if len(exp_path) > 1:
        print("Error: more than one experiments with that id found")
    df = tflog2pandas(exp_path[0])
    df = df.pivot(index="step", columns="metric")
    df_mean = df.filter(regex=(".*(custom_metrics|reward).*_mean"))
    df_mean = df_mean.rename(columns={x: x[24:-5] if "custom_metrics" in x else x[9:-5] for _,x in df.columns})
    return df_mean

def extract_data(df):
    # Convert power to kw
    compressor = df.filter(regex=".*power/compressor").to_numpy() / 1000
    total_server_fan = df.filter(regex=".*power/server_fan").to_numpy() / 1000
    total_crah_fan = df.filter(regex=".*power/crah_fan").to_numpy() / 1000
    total_server_load = df.filter(regex=".*power/total_server_load").to_numpy() / 1000
    server_loads = [df.filter(regex=f".*srv{i}/load").to_numpy() for i in range(360)]
    server_inlets = [df.filter(regex=f".*srv{i}/temp_in").to_numpy() for i in range(360)]
    energy = df.filter(regex=".*cost/energy").to_numpy() * 100
    dropped = df.filter(regex=".*job/dropped").to_numpy() 
    cold_isle = df.filter(regex=".*cost/temp_cold_isle").to_numpy() 
    reward = df.filter(regex=".*reward").to_numpy()
    crah_out_temps = [df.filter(regex=f".*crah{i}/temp_out").to_numpy() for i in range(4)]
    outdoor_temp = df.filter(regex=".*outdoor_temp").to_numpy()
                      

    time = df.index.to_numpy() / (3600 * 24)

    total_cooling_power = (compressor+total_server_fan+total_crah_fan)
    total_power = (compressor+total_server_fan+total_crah_fan+total_server_load)
    it_power = (total_server_fan+total_server_load)
    pue = total_power / it_power
    
    return {
        "time": time, 
        "total_crah_fan": total_crah_fan,
        "total_server_fan": total_server_fan,
        "total_server_load": total_server_load,
        "server_loads": server_loads,
        "server_inlets": server_inlets,
        "cooling": total_cooling_power, 
        "power": total_power, 
        "it": it_power, 
        "pue": pue, 
        "cold_isle": cold_isle, 
        "misplaced": dropped,
        "compressor": compressor,
        "reward": reward,
        "crah_out_temps": crah_out_temps,
        "outdoor_temp": outdoor_temp,
    }
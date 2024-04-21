import os
import shutil
import pandas as pd
from datetime import datetime
from collections import defaultdict
from utils.tools import *

def runway_config(AIRPORT, DATA_DIR):    
    runway_config_data = loaddata_by_suffix(AIRPORT, \
                                            "airport_config", \
                                            DATA_DIR) \
                        .sort_values(by=["timestamp"]) \
                        .set_index("timestamp")
                      
    timestamp_rwycfg = pd.Series([-1 for _ in range(8761)])
    runway_config = { "timestamp": [idx2iso(idx) for idx in range(1, 8761)],\
                      "runway_config": [] }
    curr_idx = -1
    guard_index = 0
    while True:
        next_idx = curr_idx + 1
        curr_rwy_cfg_idx = -1 if curr_idx == -1 else curr_idx
        if next_idx == len(runway_config_data):
            timestamp_rwycfg[guard_index:] = int(curr_rwy_cfg_idx)
            break
        hit_index = iso2idx(runway_config_data.index[next_idx])
        if Hour(runway_config_data.index[next_idx]):
            hit_index -= 1
        timestamp_rwycfg[guard_index : hit_index+1] = int(curr_rwy_cfg_idx)
        guard_index = hit_index + 1
        curr_idx += 1

    first_unone_idx = 0
    while True:
        if timestamp_rwycfg[first_unone_idx] != -1:
            timestamp_rwycfg[:first_unone_idx] = timestamp_rwycfg[first_unone_idx]
            timestamp_rwycfg = timestamp_rwycfg[1:]
            break
        first_unone_idx += 1
            
    for i in timestamp_rwycfg:
        runway_config["runway_config"].append(runway_config_data.iloc[i, 0])
        
    df_runway_config = pd.DataFrame(runway_config)
    df_runway_config.to_csv(os.path.join("preprocess", AIRPORT, "runway_config.csv".format(AIRPORT)), index = False)


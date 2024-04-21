import os
import shutil
import pandas as pd
from datetime import datetime
from collections import defaultdict
from utils.tools import *

config = ["arrival", "departure"]

def flight_a_d_num(AIRPORT, config_index, DATA_DIR):
    df_runway = loaddata_by_suffix(AIRPORT, "{}_runway".format(config[config_index]), DATA_DIR).drop_duplicates().set_index("gufi")
    df_mfs_runway_time = loaddata_by_suffix(AIRPORT, "mfs_runway_{}_time".format(config[config_index]), DATA_DIR).drop_duplicates().set_index("gufi")
    df_runway = df_runway[~df_runway.index.duplicated(keep='first')]
    df_mfs_runway_time = df_mfs_runway_time[~df_mfs_runway_time.index.duplicated(keep='first')]
    set_runway = set(df_runway.index.to_list())
    set_mfs_runway_time = set(df_mfs_runway_time.index.to_list())

    same_flight = set_runway & set_mfs_runway_time
    unique_flight = list(set_runway - same_flight)
    unique_mfs_flight = list(set_mfs_runway_time - same_flight)
    same_flight = list(same_flight)

    data_dict = {'gufi' : [], 'timestamp' : []}

    for gufi in same_flight:
        timestamp = average_time(df_runway.loc[gufi]["timestamp"], df_mfs_runway_time.loc[gufi]["timestamp"])
        data_dict['gufi'].append(gufi)
        data_dict['timestamp'].append(timestamp)

    for gufi in unique_flight:
        timestamp = average_time(df_runway.loc[gufi]["timestamp"])
        data_dict['gufi'].append(gufi)
        data_dict['timestamp'].append(timestamp)

    for gufi in unique_mfs_flight:
        timestamp = average_time(df_mfs_runway_time.loc[gufi]["timestamp"])
        data_dict['gufi'].append(gufi)
        data_dict['timestamp'].append(timestamp)
    
    all_flight = pd.DataFrame(data_dict).sort_values(by="timestamp").reset_index(drop=True)
    all_flight.to_csv(os.path.join("preprocess", AIRPORT,  "all_{}_flight_time.csv".format(config[config_index])), index=False)
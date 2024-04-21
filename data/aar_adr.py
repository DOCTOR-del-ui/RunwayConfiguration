import os
import shutil
import pandas as pd
from datetime import datetime
from collections import defaultdict
from utils.tools import *

config = ["arrival", "departure"]

def aar_adr(AIRPORT, config_index):
    DATA_PATH = os.path.join("preprocess", AIRPORT, "all_{}_flight_time.csv".format(config[config_index]))
    flight_time_data = pd.read_csv(DATA_PATH)
    init_timestamp = idx2timestamp(0)

    df = {"timerange" : [], "flightnumber" : []}
    sum_df = {"timestamp": [], "flightnumber" : []}

    for i in range(8760):
        left_time = datetime.fromtimestamp(init_timestamp).isoformat()
        init_timestamp += 3600
        right_time = datetime.fromtimestamp(init_timestamp).isoformat()
        df["timerange"].append("[{}-{})".format(left_time, right_time))
        df["flightnumber"].append(0)

    for _, row in flight_time_data.iterrows():
        index = iso2idx(row["timestamp"])
        if index >= 8760:
            break
        df["flightnumber"][index] += 1
        
    for idx in range(1, 8761):
        sum_df["timestamp"].append(idx2iso(idx))
        sum_df["flightnumber"].append(df["flightnumber"][idx-1])
    
    df = pd.DataFrame(df)
    df.to_csv(os.path.join("preprocess", 
                           AIRPORT,  
                           "{}_flight_traffic.csv"\
                            .format(config[config_index])), 
                            index=False)
    sum_df = pd.DataFrame(sum_df)
    sum_df.to_csv(os.path.join("preprocess", 
                               AIRPORT,  
                               "{}_timestamp_flight_traffic.csv"\
                                .format(config[config_index])), 
                                index=False)
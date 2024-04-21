import os
import shutil
import pandas as pd
from datetime import datetime
from collections import defaultdict

def loaddata_by_suffix(AIRPORT, suffix, DATA_DIR):
    return pd.read_csv(os.path.join(DATA_DIR, AIRPORT, "{}_{}.csv.bz2".format(AIRPORT, suffix)))

def gen_lamp_null_index(lamp_data):
    unique_time = lamp_data["timestamp"].unique()
    init_timestamp = int(datetime(2020, 11, 1, 0, 0, 0).timestamp())
    set_a = set()
    set_b = set(range(1, 8761))
    for isotime in unique_time:
        curr_timestamp = iso2timestamp(isotime) + 1800
        timestamp_diff = curr_timestamp - init_timestamp
        index = timestamp_diff // 3600
        set_a.add(index)
    set_c = list(set_b - set_a)
    set_c.sort()
    return set_c

def genLAMPbyTB(lamp_data, index, bias):
    curr_timestamp = idx2timestamp(index)
    if index < 1 or index > 8760:
        raise ValueError("传入的ISO格式时间不符合查询要求")
    if bias not in list(range(25)):
        raise ValueError("传入的偏移量只能为[0-24]中的整数值")
    query_isotime = timestamp2iso(curr_timestamp - 1800)
    return lamp_data[lamp_data["timestamp"] == query_isotime].iloc[bias, 2:]

def idx2ymdhw(idx):
    init_datetime = datetime(2020, 11, 1, 0, 0, 0)
    init_timestamp = int(init_datetime.timestamp())
    curr_datetime = datetime.fromtimestamp(init_timestamp + 3600 * idx)
    return [curr_datetime.year, \
            curr_datetime.month, \
            curr_datetime.day, \
            curr_datetime.hour, \
            curr_datetime.weekday()+1]

def average_time(*time_strings):
    time_objects = [datetime.fromisoformat(time_str) for time_str in time_strings]
    total_seconds = sum(time_obj.timestamp() for time_obj in time_objects)
    average_timestamp = int(total_seconds / len(time_objects))
    average_time_obj = datetime.fromtimestamp(average_timestamp)
    return average_time_obj.isoformat()

def idx2timestamp(idx):
    init_datetime = datetime(2020, 11, 1, 0, 0, 0)
    init_timestamp = int(init_datetime.timestamp())
    curr_timestamp = init_timestamp + 3600 * idx
    return int(datetime.fromtimestamp(curr_timestamp).timestamp())

def idx2iso(idx):
    return timestamp2iso(idx2timestamp(idx))

def iso2timestamp(isotime):
    return int(datetime.fromisoformat(isotime).timestamp())
    
def timestamp2iso(timestamp):
    return datetime.fromtimestamp(timestamp).isoformat()

def timestamp2idx(timestamp):
    init_datetime = datetime(2020, 11, 1, 0, 0, 0)
    init_timestamp = int(init_datetime.timestamp())
    timestamp_diff = timestamp - init_timestamp
    return int(timestamp_diff // 3600)

def iso2idx(isotime):
    return timestamp2idx(iso2timestamp(isotime))


def Hour(isotime):
    init_timestamp = idx2timestamp(0)
    time_diff = iso2timestamp(isotime) - init_timestamp
    if time_diff % 3600:
        return False
    return True
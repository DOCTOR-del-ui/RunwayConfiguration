import torch
import numpy as np
import threading
from tqdm import tqdm
from utils.tools import *
from torch.utils.data import Dataset


cloud = {
              -1: 0,
              "OV": 5,
              "BK": 4,
              "SC": 3,
              "FW": 2,
              "CL": 1
             }

lightning_prob = {
                  -1: 0,
                  "N": 1,
                  "L": 2,
                  "M": 3,
                  "H": 4
                 }

precip = {
    -1: 0,
    False: -1,
    True: 1
}


airport2idx = {
    "katl": 1,
    "kclt": 2,
    "kden": 3,
    "kdfw": 4,
    "kjfk": 5,
    "kmem": 6,
    "kmia": 7,
    "kord": 8,
    "kphx": 9,
    "ksea": 10
}

idx2rwy_config_dic = {
    1 : {
        "D_26L_27R_A_26R_27L_28" : 0,
        "D_8R_9L_A_10_8L_9R" : 1,
        "D_26L_27R_28_A_26R_27L_28" : 2,
        "D_26L_28_A_26L_28" : 3,
        "D_26L_27R_A_26L_27L_28" : 4,
        "D_26R_28_A_26R_28" : 5,
        "D_8R_9R_A_10_8L_9R" : 6,
        "D_26L_27R_A_26R_27L" : 7,
        "D_26L_27R_A_26R_27R_28" : 8,
        "D_26R_27R_A_26R_27L_28" : 9,
        "D_26L_27R_A_26L_27R_28" : 10,
        "D_10_8R_A_10_8R" : 11,
        "D_26L_27L_A_26R_27L_28" : 12,
        "D_8R_9L_A_8L_9R" : 13,
        "D_26L_27R_A_27L_28" : 14,
        "D_10_8R_9L_A_10_8L_9R" : 15,
        "D_8R_9L_A_10_9R" : 16,
        "D_10_8L_A_10_8L" : 17,
        "D_26L_28_A_26R_28" : 18,
        "D_9L_A_9R" : 19,
        "D_26L_27R_A_26R_28" : 20,
        "D_8R_9L_A_10_8R_9R" : 21,
        "D_8R_9L_A_8R_9L" : 22,
        "D_8L_9L_A_10_8L_9R" : 23,
        "D_26L_27R_A_26L_27R" : 24,
        "D_27L_28_A_27L_28" : 25,
        "other" : 26
    }
}


airport_rwy_vecpos_dic = {
   "katl" :  ["8L", "26R", "8R", "26L", "9L", "27R", "9R", "27L", "10", "28"]
}


def rwy_config_onehot(runway_config, airport):
    curr_runway_config_dict = idx2rwy_config_dic[airport2idx[airport]]
    res = [0 for _ in range(len(curr_runway_config_dict))]
    if runway_config not in curr_runway_config_dict:
        res[-1] = 1
        return res
    res[curr_runway_config_dict[runway_config]] = 1
    return res

def procLamp(curr_dec_x):
    curr_dec_x[-3] = cloud[curr_dec_x[-3]]
    curr_dec_x[-2] = lightning_prob[curr_dec_x[-2]]
    curr_dec_x[-1] = precip[curr_dec_x[-1]]
     
def rwycfg2vec(runway_config, airport):
    runway_config = runway_config.split("_")
    A_index = runway_config.index("A")
    D_index = 0
    currD = runway_config[D_index + 1 : A_index]
    currA = runway_config[A_index + 1 :]
    if airport == "katl":
        #每个跑道的配置信息用4维向量表示，一共有5条跑道
        # [8L, 26R]  => [Depature, Arrival, Depature, Arrival]
        # [8R, 26L]  => [Depature, Arrival, Depature, Arrival]
        # [9L, 27R]  => [Depature, Arrival, Depature, Arrival]
        # [9R, 27L]  => [Depature, Arrival, Depature, Arrival]
        # [10, 28]  => [Depature, Arrival, Depature, Arrival]
        rwy_vecpos_dic = airport_rwy_vecpos_dic[airport]
        init_vex = [ 0 for _ in range(4 * 5)]  
        for item in currD:
            vec_pos = 2 * rwy_vecpos_dic.index(item)
            init_vex[vec_pos] = 1000
        for item in currA:
            vec_pos = 2 * rwy_vecpos_dic.index(item) + 1
            init_vex[vec_pos] = 1000
        return init_vex



def MakeLaDataset(AIRPORT, PRECOUNT, LOOKFORWARD, DATA_DIR):
    PRE_DATA_PATH = os.path.join("preprocess", AIRPORT)
    lamp_data = loaddata_by_suffix(AIRPORT, "lamp", DATA_DIR) \
                            .sort_values(by=["timestamp","forecast_timestamp"]) \
                            .drop_duplicates().fillna(-1)                           
    aar_data = pd.read_csv(os.path.join( \
                                PRE_DATA_PATH, \
                                "arrival_timestamp_flight_traffic.csv")) \
                                .set_index("timestamp")                       
    adr_data = pd.read_csv(os.path.join( \
                                PRE_DATA_PATH, \
                                "departure_timestamp_flight_traffic.csv")) \
                                .set_index("timestamp")                                  
    rwycfg_data = pd.read_csv(os.path.join( \
                                    PRE_DATA_PATH, \
                                    "runway_config.csv")) \
                                    .set_index("timestamp")
                                    
    aar_adr_null_idx = list(range(3588, 3685))
    lamp_null_idx = gen_lamp_null_index(lamp_data)
    enc_inputs = []
    dec_inputs = []
    targets = []
    
    description = "正在制作{}机场数据集".format(AIRPORT)
    end_index = 8760-PRECOUNT-LOOKFORWARD+1
    for start_index in tqdm(range(1, end_index + 1), description):
        look_index = start_index+PRECOUNT-1
        if look_index in lamp_null_idx:
            continue
        curr_flag = True
        total_enc_x = []
        total_dec_x = []
        total_y = []
        #编码器输入
        for bias in range(PRECOUNT):
            curr_idx = start_index + bias
            if curr_idx in aar_adr_null_idx:
                curr_flag = False
                break 
            curr_enc_x =  idx2ymdhw(curr_idx)
            curr_enc_x.append(PRECOUNT - bias)
            curr_enc_x.append(aar_data.loc[idx2iso(curr_idx), "flightnumber"])
            curr_enc_x.append(adr_data.loc[idx2iso(curr_idx), "flightnumber"])
            if curr_idx in lamp_null_idx:
                curr_enc_x += [0 for _ in range(9)]
            else:
                curr_enc_x += list(genLAMPbyTB(lamp_data, curr_idx, 0))
                procLamp(curr_enc_x)
            curr_enc_x += rwycfg2vec(rwycfg_data
                                    .loc[idx2iso(curr_idx), \
                                    "runway_config"], 
                                    AIRPORT)
            total_enc_x.append(curr_enc_x)
        #解码器输入与真实跑道配置数据
        for forwardbias in range(1, LOOKFORWARD+1):
            curr_idx = look_index + forwardbias
            curr_dec_x = idx2ymdhw(curr_idx)
            curr_dec_x.append(forwardbias)
            curr_dec_x += list(genLAMPbyTB(lamp_data, look_index, forwardbias))
            procLamp(curr_dec_x)
            total_dec_x.append(curr_dec_x)
            curr_rwycfg = rwycfg_data.loc[idx2iso(curr_idx), "runway_config"]
            total_y.append(rwy_config_onehot(curr_rwycfg, AIRPORT))
        if curr_flag:
            enc_inputs.append(np.array(total_enc_x).astype(float))
            dec_inputs.append(np.array(total_dec_x).astype(float))
            targets.append(np.array(total_y).astype(float))
    return np.concatenate(enc_inputs, axis=0), \
           np.concatenate(dec_inputs, axis=0), \
           np.concatenate(targets, axis=0)
    
class LaDataset(Dataset):
    def __init__(self, enc_in_np, dec_in_np, targets_np, PRECOUNT, LOOKFORWARD):
        super(LaDataset, self).__init__()
        self.enc_inputs = torch.split(torch.tensor(enc_in_np).float(), \
                                      split_size_or_sections=PRECOUNT, dim=0)
        self.dec_inputs = torch.split(torch.tensor(dec_in_np).float(), \
                                      split_size_or_sections=LOOKFORWARD, dim=0)
        self.targets = torch.split(torch.tensor(targets_np).float(), \
                                      split_size_or_sections=LOOKFORWARD, dim=0)
        
    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, index):
        return self.enc_inputs[index], \
               self.dec_inputs[index], \
               self.targets[index]
               
    def __getitem__(self, slice):
        return self.enc_inputs[slice], \
               self.dec_inputs[slice], \
               self.targets[slice]
        



import os
from data.flight_a_d_num import flight_a_d_num
from data.aar_adr import aar_adr
from data.runway_config import runway_config


def preprocessAll(AIRPORT, DATA_DIR):
    print("正在预处理{}机场数据...".format(AIRPORT), end="\r")
    if not os.path.exists("preprocess"):
        os.mkdir("preprocess")

    if not os.path.exists(os.path.join("preprocess", AIRPORT)):
        os.mkdir(os.path.join("preprocess", AIRPORT))

    flight_a_d_num(AIRPORT, 0, DATA_DIR)
    flight_a_d_num(AIRPORT, 1, DATA_DIR)

    aar_adr(AIRPORT, 0)
    aar_adr(AIRPORT, 1)

    runway_config(AIRPORT, DATA_DIR)
    #清屏输出
    print("{}".format(" "*50), end="\r")
    print("{}机场数据处理完成".format(AIRPORT))
    


    






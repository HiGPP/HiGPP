import pandas as pd
import numpy as np




def encode_map(input_array):
    p_map={}
    length=len(input_array)
    for index, ele in zip(range(1, length + 1), input_array):
        p_map[str(ele)] = index
    return p_map


def decode_map(encode_map):
    de_map={}
    for k,v in encode_map.items():
        de_map[v]=k
    return de_map
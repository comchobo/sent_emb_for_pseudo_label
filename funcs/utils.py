import numpy as np


def get_flatten(data_list):
    lens_each_list = [len(x) for x in data_list]
    return [x for xs in data_list for x in xs], lens_each_list


def recover_flatten(data_list, lens_each_list):
    temp = []
    for lens in lens_each_list:
        temp.append(data_list[:lens])
        data_list = data_list[:lens]
    return temp

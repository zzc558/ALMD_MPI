#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:06:17 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt

def read_time(fpath, time_key):
    
    # read the raw data
    time_pl_all = []
    with open(fpath, 'rb') as fh:
        time_pl_all = np.load(fh)
    
    # dictionary to save all the processed time
    time_pl = {}
    for k in time_key:
        time_pl[k] = []
    
    # process the raw data
    # jump to the next time key if encounter the value of -1
    k = 0
    for t in time_pl_all:
        if t == -1:
            k = k + 1 if k < len(time_key)-1 else 0
        else:
            time_pl[time_key[k]].append(t)
    
    for k in time_pl.keys():
        time_pl[k] = np.array(time_pl[k], dtype=float)
        print(k, time_pl[k].shape)
    
    for k, v in time_pl.items():
        try:
            print(f"PL {k} mean time: {np.mean(np.array(v, dtype=float))}")
            print(f"PL {k} min time: {np.min(np.array(v, dtype=float))}")
            print(f"PL {k} max time: {np.max(np.array(v, dtype=float))}")
            print()
        except:
            continue
    
    return time_pl

if __name__ == "__main__":
    # list of path to all time file
    fpath = ["/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/ml_time/new_script/5ps_nvt_std1/pltime.npy",]
    
    # list of different time in the order of writing to the disk
    time_key = ['bcast', 'predict', 'gather', 'update', 'save']
    
    # load and process the raw data
    time_pl = []
    for p in fpath:
        time_pl.append(read_time(p, time_key))
    
    
    
    """
    time_means = {
        'Active learning': [np.mean(time_pl[0][k], dtype=float) for k in t_name],
        'prediction and generator kernel only': [np.mean(time_pl[1][k], dtype=float) for k in t_name],
        #'Double Processes': [np.mean(time_pl[2][k], dtype=float) for k in t_name],
        }
    
    x = np.arange(len(t_name))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained', figsize=(30,20))
    for attribute, measurement in time_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
        
    ax.set_ylabel('Time (second)', fontsize=50)
    #ax.set_title('Average Execution Time of Each Procedure by Tasks')
    ax.set_xticks(x + width, t_name)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend(loc='upper left', fontsize=50)
    #ax.set_ylim(0, 250)
    
    plt.show()
    plt.close()
    """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 04:14:09 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt

def read_time(fpath):
    time_pl_all = []
    with open(fpath, 'rb') as fh:
        time_pl_all = np.load(fh)
    
    print(np.where(time_pl_all == -1))
    time_pl_all = time_pl_all[:np.where(time_pl_all == 0)[0][0]]
    
    time_pl = {}
    t_name = ['bcast', 'predict', 'gather', 'update']
    #t_name = ['bcast', 'predict', 'gather']
    for k in t_name:
        time_pl[k] = []
    
    k = 0
    for t in time_pl_all:
        if t == -1:
            k = k + 1 if k < 3 else 0
        else:
            time_pl[t_name[k]].append(t)
    
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
    fpath = ["../results/al_time/init_opt/pltime2.npy", "../results/al_time/retrained_model_no_ml/pltime.npy",]
    time_pl = []
    for p in fpath:
        time_pl.append(read_time(p))
    
    t_name = ['bcast', 'predict', 'gather']
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

    
    
    
    #fig = plt.figure()
    #plt.hist(time_pl['bcast'], bins=100)
    #plt.xlabel("Time [s]")
    #plt.ylabel("Count")
    #plt.title("Time Distribution of PL Receiving Message from MG")
    #plt.show()
    #plt.close()
    
    #plt.plot(np.array(time_pl['bcast'][:200000], dtype=float))
    #plt.show()
    #plt.close()
    
    #t_name = ['bcast', 'predict', 'gather']
    #fig, ax = plt.subplots(figsize=(12,10))
    #keys = ["Receive Coordinates", "Prediction", "Send Predictions"]
    #time = [np.mean(time_pl[k], dtype=float) for k in t_name]
    #bar_colors = ["#1F77B4", "#D40000", "#1F77B4"]
    #ax.bar(keys, time, color=bar_colors)
    #ax.set_ylabel("Time [s]", fontsize=20)
    #ax.set_title("Time Cost for PL and MD Kernal", fontsize=20)
    #plt.show()
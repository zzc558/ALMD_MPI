#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 05:23:55 2023

@author: chen
"""

import matplotlib.pyplot as plt
import numpy as np
import json

if __name__ == "__main__":
    data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/retrain_history/n5000_retrain_20231001/retrain_history0.json"
    
    with open(data_path, "r") as fh:
        hist = json.load(fh)
    
    print(hist.keys())
    
    energy_mae_train = np.array(hist["energy_mean_absolute_error"], dtype=float)
    energy_mae_val = np.array(hist["val_energy_mean_absolute_error"], dtype=float)
    #force_mae_train = np.array(hist["force_mean_absolute_error"], dtype=float)
    #force_mae_val = np.array(hist["val_force_mean_absolute_error"], dtype=float)
    
    print(energy_mae_train.shape)
    print(energy_mae_val.shape)
    #print(force_mae_train.shape)
    #print(force_mae_val.shape)
    
    fig, ax1 = plt.subplots(figsize=(30, 20))
    lns1 = ax1.plot(np.arange(1, energy_mae_train.shape[0]+1), energy_mae_train, linestyle='-', color='#1f77b4', label='Training energy')
    lns2 = ax1.plot(np.arange(10, 10*energy_mae_val.shape[0]+1, 10), energy_mae_val, linestyle='--', color='#1f77b4', label='Validation energy')
    ax1.set_ylabel("Energy mean absolute error (eV)", fontsize=50)
    ax1.set_xlabel("Number of epoch", fontsize=50)
    ax1.tick_params(axis='x', labelsize=30)
    ax1.tick_params(axis='y', labelsize=30)
    #ax1.legend(fontsize=50, markerscale=5)
    
    #ax2 = ax1.twinx()
    #lns3 = ax2.plot(np.arange(1, force_mae_train.shape[0]+1), force_mae_train, linestyle='-', color='#ff7f0e', label='Training force')
    #lns4 = ax2.plot(np.arange(10, force_mae_train.shape[0]+1, 10), force_mae_val, linestyle='--', color='#ff7f0e', label='Validation force')
    #ax2.set_ylabel("Force mean absolute error (eV)", fontsize=50)
    #ax2.tick_params(axis='y', labelsize=30)
    
    #lns = lns1 + lns2 + lns3 + lns4
    #labs = [l.get_label() for l in lns]
    #leg = ax1.legend(lns, labs, loc=0, fontsize=50)
    # set the linewidth of each legend object
    #for legobj in leg.legendHandles:
    #    legobj.set_linewidth(5.0)

    
    plt.show()
    plt.close()

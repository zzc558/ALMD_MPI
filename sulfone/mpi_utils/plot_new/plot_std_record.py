#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:04:33 2023

@author: chen

Function:
    plot the energy prediction std for geometries with std > 0.5 for at least one prediction. 
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/std/retrain_1ns_nvt/std_record.npy"
    
    # load std record
    std_record = []
    with open(data_path, 'rb') as fh:
        while True:
            try:
                std_record += np.load(fh).tolist()
            except:
                break
    std_record = np.array(std_record, dtype=float)
    print(std_record.shape)

    print(np.max(std_record))
    print(np.min(std_record))
    print(np.count_nonzero((std_record>2.0).any(axis=1)))
    
    # plot the trend of std for all electornic states
    fig, ax = plt.subplots(figsize=(30,20))
    for i in range(0, std_record.shape[1]):
        ax.scatter(np.arange(0, std_record.shape[0]), std_record[:,i], label=f"S{i}")
    ax.set_ylabel('Energy prediction STD (eV)', fontsize=50)
    ax.legend(loc='upper right', fontsize=50)
    ax.tick_params(axis='both', labelsize=30)
    plt.show()
    plt.close()
    
    # plot trend of std for each individual state
    for i in range(0, std_record.shape[1]):
        fig, ax = plt.subplots(figsize=(30,20))
        ax.scatter(np.arange(0, std_record.shape[0]), std_record[:,i], label=f"S{i}")
        ax.set_ylabel('Energy prediction STD (eV)', fontsize=50)
        ax.legend(loc='upper right', fontsize=50)
        ax.tick_params(axis='both', labelsize=30)
        ax.set_ylim(-0.1, 2.5)
        plt.show()
        plt.close()
        
    # plot heatmap for trend of std for each individual state
    for i in range(0, std_record.shape[1]):
        fig, ax = plt.subplots(figsize=(30,20))
        h = ax.hist2d(np.arange(0, std_record.shape[0]), std_record[:,i], bins=100, cmap='autumn')
        fig.colorbar(h[3], ax=ax)
        ax.set_ylabel('Energy prediction STD (eV)', fontsize=50)
        ax.tick_params(axis='both', labelsize=30)
        ax.set_ylim(-0.1, 2.5)
        plt.show()
        plt.close()

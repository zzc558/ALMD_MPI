#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:18:36 2023

@author: chen

Function:
    Plot bond length vs trajectory steps.
    
Bond length calculated by data_process/calc_bond_len_ml.py on Horeka.
"""

import numpy as np
import matplotlib.pyplot as plt


BOND_LIMIT = {'OS': 2.5, 'CS': 2.5, 'CC': 2.3, 'CH': 6.0}

if __name__ == "__main__":
    #data_file = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/ml_traj/new_script/5ps_nvt_std1/s_biph_bond_len.npy"
    data_file = "/home/chen/Documents/blueOLED/BlueOledData/src/s_ar_len_s6.npy"
    
    with open(data_file, "rb") as fh:
        bond_len = np.load(fh)
        
    i_non_zero = np.count_nonzero(bond_len, axis=-1)
    print(i_non_zero)
        
    fig, ax = plt.subplots(figsize=(20, 15))
    for i in range(0, bond_len.shape[0]):
        ax.plot(np.arange(1, bond_len[i][:i_non_zero[i]].shape[0]+1), bond_len[i][:i_non_zero[i]], linewidth=2.5, color='#1f77b4')
    ax.set_ylabel("Bond length (angstrom)", fontsize=40)
    ax.set_xlabel("Time step", fontsize=40)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.set_xlim(1.40, 4.0)
    #ax.set_ylim(1.40, 4.0)
    plt.show()
    plt.close()
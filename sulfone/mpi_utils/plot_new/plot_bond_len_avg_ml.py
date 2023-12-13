#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:16:51 2023

@author: chen

Function:
    Plot average bond length.
    
Bond length calculated by data_process/calc_bond_len_ml.py and data_process/calc_bond_len_new.py on Horeka.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


BOND_LIMIT = {'OS': 2.5, 'CS': 2.5, 'CC': 2.3, 'CH': 6.0}

if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/ml_traj/new_script/5ps_nvt_std1_init_alldata"
    
    for f in os.listdir(data_dir):
        if "bond_len" in f:
            print(f)
            with open(os.path.join(data_dir, f), "rb") as fh:
                bond_len = np.load(fh)
                
            print(bond_len.shape)
            print(np.count_nonzero(bond_len, axis=-1))
            
            i_non_zero = np.count_nonzero(bond_len, axis=-1)
            #for i in range(0, bond_len.shape[0]):
            #    bond_len[i][bond_non_zero[i]:] = bond_len[i][bond_non_zero[i]-1]

            
            

            # trajectories terminated due to high STD: counting stops at where trajectory stops
            # trajectories terminated due to abnormal bond length: the last bond length record is copied for the rest time steps
            bond_len_selected = np.zeros_like(bond_len, dtype=float)
            for i in range(0, bond_len.shape[0]):
                bond_len_selected[i, :i_non_zero[i]] = bond_len[i, :i_non_zero[i]]
                if bond_len[i, i_non_zero[i]-1] >= BOND_LIMIT['CS']:
                    bond_len_selected[i, i_non_zero[i]:] = bond_len[i, i_non_zero[i]-1]
            #bond_len_selected = []
            #for i in range(0, bond_len.shape[0]):
            #    if bond_len[i, i_non_zero[i]-1] < BOND_LIMIT['CS']:
            #        bond_len_selected.append(bond_len[i, :i_non_zero[i]])
            #    else:
            #        l = np.empty((bond_len.shape[1],), dtype=float)
            #        l[:i_non_zero[i]] = bond_len[i, :i_non_zero[i]]
            #        l[i_non_zero[i]:] = bond_len[i, i_non_zero[i]-1]
            #        bond_len_selected.append(l)

            # calculate average bond length
            bond_len_avg = np.empty((bond_len_selected.shape[1],), dtype=float)
            for i in range(0, bond_len_selected.shape[1]):
                bond_len_avg[i] = np.sum(bond_len_selected[:, i]) / np.count_nonzero(bond_len_selected[:, i])
            
            #bond_len_avg = np.average(bond_len, axis=0)
            #print(bond_len_avg.shape)

            #bond_len_avg = np.empty((bond_len.shape[1],), dtype=float)
            #for i in range(0, bond_len_avg.shape[0]):
            #    len_sum = 0
            #    len_count = 0
            #    for l in bond_len_selected:
            #        if len(l) > i:
            #            len_sum += l[i]
            #            len_count += 1
            #    if len_count == 0:
            #        break
            #    bond_len_avg[i] = len_sum / len_count
            
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.plot(np.arange(1, bond_len_avg.shape[0]+1), bond_len_avg, linewidth=1.5)
            ax.set_ylabel("Average bond length (angstrom)", fontsize=40)
            ax.set_xlabel("Time step", fontsize=40)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            #ax.set_xlim(1.40, 4.0)
            #ax.set_ylim(1.40, 4.0)
            plt.show()
            plt.close()
            
        
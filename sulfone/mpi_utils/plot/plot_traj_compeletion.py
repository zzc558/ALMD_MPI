#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:38:32 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
import os, pickle


BOND_INDEX = {
                'OS': [[0, 1], [0, 2]], 'CS': [[0, 3], [0, 13]],\
                'CC': [[3, 4], [3, 5], [4, 6], [5, 8], [6, 10], [6, 34],\
                       [8, 10], [13, 14], [13, 15], [14, 16], [15, 18],\
                       [16, 20], [18, 20], [20, 23], [23, 24], [23, 25],\
                       [24, 26], [25, 28], [26, 30], [28, 30]],\
                'CH': [[4, 7], [5, 9], [8, 11], [10, 12], [14, 17], [15, 19],\
                       [16, 21], [18, 22], [24, 27], [25, 29], [26, 31],\
                       [28, 32], [30, 33], [34, 35], [34, 36], [34, 37]]
                    }
    
BOND_LIMIT = {'OS': 2.5, 'CS': 2.75, 'CC': 2.3, 'CH': 6.0}

def calc_bond_length(coord):
    c1 = np.expand_dims(coord, axis=0)
    c2 = np.expand_dims(coord, axis=1)
    dist_matrix = np.sqrt(np.sum(np.square(c1-c2), axis=-1))
    bond_length = {}
    for b, idx in BOND_INDEX.items():
        bond_length[b] = [dist_matrix[i,j] for (i,j) in idx]
    return bond_length

def check_bond_length(coord):
    bond_lengths = calc_bond_length(coord)
    for b, l in BOND_LIMIT.items():
        if (np.array(bond_lengths[b], dtype=float) > l).any():
            return False, b
    return True, None

if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    
    max_length = 200
    n_complete = 0
    n_high_std = 0
    n_bond_break = 0
    for f in os.listdir(data_dir):
        #if f.endswith(".npy"):
        if "trajdata_organize" in f:
            with open(os.path.join(data_dir, f), 'rb') as fh:
                coord = np.load(fh)
            #print(coord.shape)
            traj_length = np.count_nonzero(coord[:,:,0,0], axis=1)
            for i in range(0, coord.shape[0]):
                #print(traj_length[i])
                if traj_length[i] == max_length:
                    n_complete += 1
                elif check_bond_length(coord[i][traj_length[i]-1])[0]:
                    n_high_std += 1
                else:
                    n_bond_break += 1
                    
    print(n_complete)
    print(n_high_std)
    print(n_bond_break)
            
                
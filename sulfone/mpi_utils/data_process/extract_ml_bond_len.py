#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 07:42:09 2023

@author: chen

Function:
    Calculate bond length from coordinates of trajectories.
    
Coordinate data from read_traj.py on Horeka.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/ml_traj/ml_5ps_nvt_complete"
    save_file = ""
    
    target_bond = np.zeros((1, 200), dtype=float)
    atom_index_0 = 0
    atom_index_1 = 3
    
    for f in os.listdir(data_dir):
        if "coord" in f:
            with open(os.path.join(data_dir, f), "rb") as fh:
                traj_coord = np.load(fh)[1:]
            
            traj_len = np.count_nonzero(traj_coord[:,:,0,0], axis=-1)
            traj_coord = traj_coord[traj_len>0]
            for i in range(0, len(traj_coord)):
                traj_coord[i][traj_len[i]:] = traj_coord[i][traj_len[i]-1]

            atom_coord_0 = traj_coord[:,:,atom_index_0]
            atom_coord_1 = traj_coord[:,:,atom_index_1]
            bond_len = np.linalg.norm(atom_coord_0 - atom_coord_1, axis=-1)
            
            target_bond = np.append(target_bond, bond_len, axis=0)
            
    print(target_bond[1:].shape)
            

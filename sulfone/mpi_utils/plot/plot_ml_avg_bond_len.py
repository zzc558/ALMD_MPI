#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:54:39 2023

@author: chen

Function:
    plot average bond length change for MLMD data
"""

import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np

if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/ml_traj/ml_5ps"
    state_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/ml_traj/ml_5ps/state"
    
    len_s_ar = []
    len_s_biph = []
    
    init_state = 1
    
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            with open(os.path.join(data_dir, f), 'rb') as fh:
                coords = np.load(fh)
            c1_atom_coord = coords[:,:,3]
            c2_atom_coord = coords[:,:,13]
            s_atom_coord = coords[:,:,0]
            
            file_index = int(f.split('_')[-1].split('.')[0])
            state_file = f"trajdata_states_{file_index}.npy"
            if not os.path.exists(os.path.join(state_dir, state_file)):
                continue
            
            with open(os.path.join(state_dir, state_file), 'rb') as fh:
                states = np.load(fh).astype(int)
                
            
            
            len_s_ar.append(np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1))
            len_s_biph.append(np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:24:52 2023

@author: chen
"""

import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np
import os, pickle

if __name__ == "__main__":
    al_data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/new_training_set0.npy"
    init_data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/results/eg_model_n5000_e1000/eg"
    
    with open(al_data_path, 'rb') as fh:
        coords = np.load(fh)
    print(coords.shape)

    c1_atom_coord = coords[:,3]
    c2_atom_coord = coords[:,13]
    s_atom_coord = coords[:,0]
    len_s_ar = np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1)
    len_s_biph = np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1)
    print(len_s_ar.shape)
    
    with open(os.path.join(init_data_path, "index/train_val_idx_v0.npy"), 'rb') as fh:
        i_train = np.load(fh)
    print(i_train.shape)
    with open(os.path.join(init_data_path, "data_x"), 'rb') as fh:
        coords_init = pickle.load(fh)[i_train]
    print(coords_init.shape)
    
    c1_atom_coord_init = coords_init[:,3]
    c2_atom_coord_init = coords_init[:,13]
    s_atom_coord_init = coords_init[:,0]
    len_s_ar_init = np.linalg.norm(s_atom_coord_init - c1_atom_coord_init, axis=-1)
    len_s_biph_init = np.linalg.norm(s_atom_coord_init - c2_atom_coord_init, axis=-1)
    print(len_s_ar_init.shape)
    
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.scatter(len_s_ar, len_s_biph, color='cyan', s=0.5, label='Al dataset')
    #ax.scatter(len_s_ar_init, len_s_biph_init, color='red', s=0.5, label='Init dataset')
    ax.set_title("Sulfur Carbon Length")
    ax.set_xlabel("S-Ar Bond Length")
    ax.set_ylabel("S-Biphenyl Bond Length")
    ax.set_xlim(1.0, 4.0)
    ax.set_ylim(1.0, 4.0)
    ax.legend()
    plt.show()
    plt.close()
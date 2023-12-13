#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:42:45 2023

@author: chen
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np
import os, pickle

if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    len_s_ar = np.zeros((1, 200), dtype=float)
    len_s_biph = np.zeros((1, 200), dtype=float)
    eng_sel = np.zeros((1, 200, 7), dtype=float)
    state_sel = np.zeros((1, 200), dtype=float)
    #eng_s_ar = np.zeros((1, 200, 7), dtype=float)
    #eng_s_biph = np.zeros((1, 200, 7), dtype=float)
    #state_s_ar = np.zeros((1, 200), dtype=float)
    #state_s_biph = np.zeros((1, 200), dtype=float)

    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            with open(os.path.join(data_dir, f), 'rb') as fh:
                coords = np.load(fh)[1:]
                engs = np.load(fh)[1:]
                states = np.load(fh)[1:]
            c1_atom_coord = coords[:,:,3]
            c2_atom_coord = coords[:,:,13]
            s_atom_coord = coords[:,:,0]
            l1 = np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1)
            l2 = np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1)
            #l1_idx = list(set(np.where(l1>3.0)[0]))
            #l2_idx = list(set(np.where(l2>3.0)[0]))
            idx_sel = list(set(np.where(l1>3.0)[0]).union(set(np.where(l2>3.0)[0])))
            len_s_ar = np.append(len_s_ar, l1[idx_sel], axis=0)
            len_s_biph = np.append(len_s_biph, l2[idx_sel], axis=0)
            eng_sel = np.append(eng_sel, engs[idx_sel], axis=0)
            state_sel = np.append(state_sel, states[idx_sel], axis=0)
            #eng_s_ar = np.append(eng_s_ar, engs[l1_idx], axis=0)
            #eng_s_biph = np.append(eng_s_biph, engs[l2_idx], axis=0)
            #state_s_ar = np.append(state_s_ar, states[l1_idx], axis=0)
            #state_s_biph = np.append(state_s_biph, states[l2_idx], axis=0)
    
    len_s_ar = len_s_ar[1:]
    len_s_biph = len_s_biph[1:]
    eng_sel = eng_sel[1:]
    state_sel = state_sel[1:]
    #eng_s_ar = np.array(eng_s_ar, dtype=float)[1:]
    #eng_s_biph = np.array(eng_s_biph, dtype=float)[1:]
    #state_s_ar = np.array(state_s_ar, dtype=float)[1:]
    #state_s_biph = np.array(state_s_biph, dtype=float)[1:]
    
    print(len_s_ar.shape)
    print(len_s_biph.shape)
    print(eng_sel.shape)
    print(state_sel.shape)
    
    idx_sel = np.min(np.where(len_s_biph == 0)[-1])
    print(idx_sel)
    print(len_s_ar[:,:idx_sel].shape)
    print(len_s_biph[:,:idx_sel].shape)
    print(eng_sel[:,:idx_sel,0].shape)
    
    fig, ax = plt.subplots(figsize=[10, 10], subplot_kw=dict(projection='3d'))
    surf = ax.plot_surface(len_s_ar[:,:idx_sel], len_s_biph[:,:idx_sel], eng_sel[:,:idx_sel,0], cmap='viridis', edgecolor='none')
    plt.show()
    plt.close()
        
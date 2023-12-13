#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:17:19 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os


if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    coords = np.empty((1, 200, 38, 3), dtype=float)
    engs = np.empty((1, 200, 7), dtype=float)
    states = np.empty((1, 200), dtype=float)
    len_s_ar = []
    len_s_biph = []
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            with open(os.path.join(data_dir, f), 'rb') as fh:
                coords = np.append(coords, np.load(fh)[1:], axis=0)
                engs = np.append(engs, np.load(fh)[1:], axis=0)
                states = np.append(states, np.load(fh)[1:], axis=0)
            c1_atom_coord = coords[1:,:,3]
            c2_atom_coord = coords[1:,:,13]
            s_atom_coord = coords[1:,:,0]
            len_s_ar.append(np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1))
            len_s_biph.append(np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1))
        break
    coords = coords[1:]
    engs = engs[1:]
    states = states[1:]
    
    coords = coords.reshape(coords.shape[0]*coords.shape[1], 38, 3)
    engs = engs.reshape(engs.shape[0]*engs.shape[1], 7)
    states = states.flatten().astype(int)
    engs_state = np.expand_dims(np.array([engs[i][states[i]] for i in range(0, states.shape[0])], dtype=float), axis=1)
    len_s_ar = np.array(len_s_ar, dtype=float).flatten()
    len_s_biph = np.array(len_s_biph, dtype=float).flatten()
    
    i_nonzero = np.where(len_s_ar>0)[0]
    len_s_ar = len_s_ar[i_nonzero]
    len_s_biph = len_s_biph[i_nonzero]
    engs_state = engs_state[i_nonzero]
    
    print(coords.shape)
    print(engs.shape)
    print(states.shape)
    print(engs_state.shape)
    print(len_s_ar.shape)
    print(len_s_biph.shape)
    
    colormap = plt.get_cmap('viridis')
    norm = Normalize(vmin=engs_state.min(), vmax=engs_state.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.scatter(len_s_ar[:10000], len_s_biph[:10000], c=engs_state[:10000], cmap=colormap, norm=norm)
    ax.set_ylabel("S-Biphenyl Bond Length", fontsize=20)
    ax.set_xlabel("S-Ar Bond Length", fontsize=20)
    plt.colorbar(sm, label='Color Value')
    plt.show()
    plt.close()
    
    #fig, ax = plt.subplots(figsize=[10, 10], subplot_kw=dict(projection='3d'))
    #surf = ax.plot_surface(len_s_ar[:10000], len_s_biph[:10000], engs_state[:10000], cmap='viridis', edgecolor='none')
    #plt.show()
    #plt.close()
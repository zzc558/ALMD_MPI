#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 23:45:34 2023

@author: chen
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np
import os, pickle

def read_al():
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    len_s_ar = np.zeros((1, 200), dtype=float)
    len_s_biph = np.zeros((1, 200), dtype=float)
    eng_s_ar = np.zeros((1, 200, 7), dtype=float)
    eng_s_biph = np.zeros((1, 200, 7), dtype=float)
    state_s_ar = np.zeros((1, 200), dtype=float)
    state_s_biph = np.zeros((1, 200), dtype=float)

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
            l1_idx = list(set(np.where(l1>3.0)[0]))
            l2_idx = list(set(np.where(l2>3.0)[0]))
            len_s_ar = np.append(len_s_ar, l1[l1_idx], axis=0)
            len_s_biph = np.append(len_s_biph, l2[l2_idx], axis=0)
            eng_s_ar = np.append(eng_s_ar, engs[l1_idx], axis=0)
            eng_s_biph = np.append(eng_s_biph, engs[l2_idx], axis=0)
            state_s_ar = np.append(state_s_ar, states[l1_idx], axis=0)
            state_s_biph = np.append(state_s_biph, states[l2_idx], axis=0)
    
    len_s_ar = len_s_ar[1:]
    len_s_biph = len_s_biph[1:]
    eng_s_ar = eng_s_ar[1:]
    eng_s_biph = eng_s_biph[1:]
    state_s_ar = state_s_ar[1:]
    state_s_biph = state_s_biph[1:]
    
    with open(os.path.join(data_dir, "s_ar.npy"), 'wb') as fh:
        np.save(fh, len_s_ar)
        np.save(fh, eng_s_ar)
        np.save(fh, state_s_ar)
        
    with open(os.path.join(data_dir, "s_biph.npy"), 'wb') as fh:
        np.save(fh, len_s_biph)
        np.save(fh, eng_s_biph)
        np.save(fh, state_s_biph)
    
    #print(len_s_ar.shape)
    #print(len_s_biph.shape)
    #print(eng_s_ar.shape)
    #print(eng_s_biph.shape)
    #print(state_s_ar.shape)
    #print(state_s_biph.shape)
    
    #len_s_ar = len_s_ar[0]
    #eng_s_ar = eng_s_ar[0]
    #state_s_ar = state_s_ar[0]
    
    #len_s_ar = len_s_ar.flatten()
    #state_s_ar = state_s_ar.flatten().astype(int)
    #eng_s_ar = eng_s_ar.reshape(eng_s_ar.shape[0]*eng_s_ar.shape[1], 7)
    #eng_s_ar = np.array([eng_s_ar[i][state_s_ar[i]] for i in range(0, state_s_ar.shape[0])], dtype=float)
    #eng_s_ar = eng_s_ar[:,0]
    #idx = np.where(len_s_ar>0)[0]
    #len_s_ar = len_s_ar[idx]
    #eng_s_ar = eng_s_ar[idx]
    #print(len_s_ar.shape)
    #print(eng_s_ar.shape)
    
    #fig, ax = plt.subplots(figsize=[10, 10])
    #ax.scatter(len_s_ar, eng_s_ar)
    #plt.show()
    #plt.close()
    
if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/results/eg_model_n5000_e1000/eg"
    with open(os.path.join(data_dir, "data_x"), 'rb') as fh:
        coords = pickle.load(fh)
    with open(os.path.join(data_dir, "data_y"), 'rb') as fh:
        engs, forces = pickle.load(fh)
        
    print(coords.shape)
    print(engs.shape)
    print(forces.shape)
    
    states = np.where(forces[:,:,0,0] != 0)[1]
    
    c1_atom_coord_init = coords[:,3]
    c2_atom_coord_init = coords[:,13]
    s_atom_coord_init = coords[:,0]
    len_s_ar = np.linalg.norm(s_atom_coord_init - c1_atom_coord_init, axis=-1)
    len_s_biph = np.linalg.norm(s_atom_coord_init - c2_atom_coord_init, axis=-1)
    l1_idx = list(set(np.where(len_s_ar>2.5)[0]))
    l2_idx = list(set(np.where(len_s_biph>3.0)[0]))
    
    print(np.max(len_s_ar))
    
    len_s_ar = len_s_ar[l1_idx]
    eng_s_ar = engs[l1_idx]
    s_s_ar = states[l1_idx]
    
    len_s_biph = len_s_biph[l2_idx]
    eng_s_biph = engs[l2_idx]
    s_s_biph = states[l2_idx]
    
    print(len_s_ar.shape)
    print(eng_s_ar.shape)
    print(s_s_ar.shape)
    print(len_s_biph.shape)
    print(eng_s_biph.shape)
    print(s_s_biph.shape)
    
    with open(os.path.join("/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data", "s_biph_init.npy"), "wb") as fh:
        np.save(fh, len_s_biph)
        np.save(fh, eng_s_biph)
        np.save(fh, s_s_biph)

    
    
    
    
    
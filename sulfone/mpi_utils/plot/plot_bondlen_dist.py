#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:48:38 2023

@author: chen
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np
import os, pickle

if __name__ == "__main__":
    #al_data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/traj_data_1_1000.npy"
    #init_data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/results/eg_model_n5000_e1000/eg"
    
    #with open(al_data_path, 'rb') as fh:
    #    coords = np.load(fh)[1:]
    #    engs = np.load(fh)[1:]
    #    states = np.load(fh)[1:]
    
    #print(coords.shape)
    #print(engs.shape)
    #print(states.shape)
    
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    len_s_ar = []
    len_s_biph = []
    len_c_h1 = []
    len_c_h2 = []
    len_s_o1 = []
    len_s_o2 = []
    imax = 0
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            with open(os.path.join(data_dir, f), 'rb') as fh:
                coords = np.load(fh)[1:]
            c1_atom_coord = coords[:,:,3]
            c2_atom_coord = coords[:,:,13]
            s_atom_coord = coords[:,:,0]
            
            s_o1_coord = coords[:,:,1]
            s_o2_coord = coords[:,:,2]
            
            c_atom_coord = coords[:,:,34]
            h1_atom_coord = coords[:,:,35]
            h2_atom_coord = coords[:,:,36]
            
            len_s_ar.append(np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1))
            len_s_biph.append(np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1))
            
            len_s_o1.append(np.linalg.norm(s_atom_coord - s_o1_coord, axis=-1))
            len_s_o2.append(np.linalg.norm(s_atom_coord - s_o2_coord, axis=-1))
            
            len_c_h1.append(np.linalg.norm(c_atom_coord - h1_atom_coord, axis=-1))
            len_c_h2.append(np.linalg.norm(c_atom_coord - h2_atom_coord, axis=-1))
            
            imax = coords.shape[0] if imax < coords.shape[0] else imax
            break
        
    print(len_s_ar[0].shape)
    
    init_data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/results/eg_model_n5000_e1000/eg"
    with open(os.path.join(init_data_path, "index/train_val_idx_v0.npy"), 'rb') as fh:
        i_train = np.load(fh)
    with open(os.path.join(init_data_path, "data_x"), 'rb') as fh:
        coords_init = pickle.load(fh)[i_train]
    
    c1_atom_coord_init = coords_init[:,3]
    c2_atom_coord_init = coords_init[:,13]
    s_atom_coord_init = coords_init[:,0]
    
    c_atom_coord_init = coords[:,34]
    h1_atom_coord_init = coords[:,35]
    h2_atom_coord_init = coords[:,36]
    
    len_s_ar_init = np.linalg.norm(s_atom_coord_init - c1_atom_coord_init, axis=-1)
    len_s_biph_init = np.linalg.norm(s_atom_coord_init - c2_atom_coord_init, axis=-1)
    len_c_h1_init = np.linalg.norm(c_atom_coord_init - h1_atom_coord_init, axis=-1)
    len_c_h2_init = np.linalg.norm(c_atom_coord_init - h2_atom_coord_init, axis=-1)
    
    #colors = mpl.colormaps['Blues'](range(imax, 0, -1))
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_s_ar)):
        for j in range(0, len_s_ar[i].shape[0]):
            ax.plot(len_s_ar[i][j][(np.where(len_s_ar[i][j]>0))[0]], len_s_biph[i][j][(np.where(len_s_biph[i][j]>0))[0]], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    ax.scatter(len_s_ar_init, len_s_biph_init, color='#ff7f0e', s=8, label='Init dataset')
    #ax.set_title("Sulfur Carbon Length", fontsize=20)
    ax.set_xlabel("S-PhMe bond length (angstrom)", fontsize=80)
    ax.set_ylabel("S-Biphenyl bond length (angstrom)", fontsize=80)
    #ax.set_xlim(1.0, 4.0)
    #ax.set_ylim(1.0, 4.0)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.legend()
    plt.show()
    plt.close()
    
    
    # plot S-O2 vs S-Biphenyl
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_s_o2)):
        for j in range(0, len_s_o2[i].shape[0]):
            ax.plot(len_s_o2[i][j][(np.where(len_s_o2[i][j]>0))[0]], len_s_biph[i][j][(np.where(len_s_biph[i][j]>0))[0]], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    #ax.scatter(len_c_h1_init, len_c_h2_init, color='red', s=0.5, label='Init dataset')
    #ax.set_title("Carbon Hydrogen Bond Length of Methyl Group", fontsize=20)
    ax.set_xlabel("C-O2 bond length (angstrom)", fontsize=80)
    ax.set_ylabel("C-Biphenyl bond length (angstrom)", fontsize=80)
    #ax.set_xlim(1.0, 4.0)
    #ax.set_ylim(1.0, 4.0)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.legend()
    plt.show()
    plt.close()
    
    # plot S-O1 vs S-Ar
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_s_o1)):
        for j in range(0, len_s_o1[i].shape[0]):
            ax.plot(len_s_o1[i][j][(np.where(len_s_o1[i][j]>0))[0]], len_s_ar[i][j][(np.where(len_s_ar[i][j]>0))[0]], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    #ax.scatter(len_c_h1_init, len_c_h2_init, color='red', s=0.5, label='Init dataset')
    #ax.set_title("Carbon Hydrogen Bond Length of Methyl Group", fontsize=20)
    ax.set_xlabel("C-O1 bond length (angstrom)", fontsize=80)
    ax.set_ylabel("C-PhMe bond length (angstrom)", fontsize=80)
    #ax.set_xlim(1.0, 4.0)
    #ax.set_ylim(1.0, 4.0)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.legend()
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_c_h1)):
        for j in range(0, len_c_h1[i].shape[0]):
            ax.plot(len_c_h1[i][j][(np.where(len_c_h1[i][j]>0))[0]], len_c_h2[i][j][(np.where(len_c_h2[i][j]>0))[0]], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    #ax.scatter(len_c_h1_init, len_c_h2_init, color='red', s=0.5, label='Init dataset')
    #ax.set_title("Carbon Hydrogen Bond Length of Methyl Group", fontsize=20)
    ax.set_xlabel("C-H1 bond length (angstrom)", fontsize=80)
    ax.set_ylabel("C-H2 bond length (angstrom)", fontsize=80)
    #ax.set_xlim(1.0, 4.0)
    #ax.set_ylim(1.0, 4.0)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.legend()
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_c_h1)):
        for j in range(0, len_c_h1[i].shape[0]):
            ax.plot(len_c_h1[i][j][(np.where(len_c_h1[i][j]>0))[0]], len_s_biph[i][j][(np.where(len_s_biph[i][j]>0))[0]], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    #ax.scatter(len_c_h1_init, len_c_h2_init, color='red', s=0.5, label='Init dataset')
    #ax.set_title("Carbon Hydrogen Bond Length of Methyl Group", fontsize=20)
    ax.set_ylabel("S-Biphenyl bond length (angstrom)", fontsize=50)
    ax.set_xlabel("C-H bond length (angstrom)", fontsize=50)
    #ax.set_xlim(1.0, 4.0)
    #ax.set_ylim(1.0, 4.0)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    #ax.legend()
    plt.show()
    plt.close()
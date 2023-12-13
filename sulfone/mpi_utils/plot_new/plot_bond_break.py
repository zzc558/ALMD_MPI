#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:36:42 2023

@author: chen

Function:
    1. Identify the bond with abnormal length that leads to trajectory termination.
    2. Plot S-Biphenyl vs S-Aromatic bond length

Coordinates of trajectories terminated with abnormal bond come from data_process/read_coord_termin.py on Horeka.
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

BOND_LIMIT = {'OS': 2.5, 'CS': 4.0, 'CC': 2.3, 'CH': 6.0}

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
    data_file = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain_1ns/new_script/bond_breaking_traj.pickle"
    
    with open(data_file, "rb") as fh:
        break_trajs = pickle.load(fh)
        
    # identify the abnormal bond that leads to the termination of corresponding trajectory
    for traj in break_trajs:
        coord_last = traj[-1]
        print(check_bond_length(coord_last))
        
    # plot S-biph vs S-Ar
    len_s_ar = []
    len_s_biph = []
    for traj in break_trajs:
        c1_atom_coord = traj[:,3]
        c2_atom_coord = traj[:,13]
        s_atom_coord = traj[:,0]
        
        len_s_ar.append(np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1))
        len_s_biph.append(np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1))
        
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_s_ar)):
        ax.plot(len_s_ar[i], len_s_biph[i], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    ax.set_xlabel("S-PhMe bond length (angstrom)", fontsize=80)
    ax.set_ylabel("S-Biphenyl bond length (angstrom)", fontsize=80)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    ax.set_xlim(1.40, 4.0)
    ax.set_ylim(1.40, 4.0)
    plt.show()
    plt.close()
    
    # plot S-O vs S-O
    len_s_o1 = []
    len_s_o2 = []
    for traj in break_trajs:
        o1_atom_coord = traj[:,1]
        o2_atom_coord = traj[:,2]
        s_atom_coord = traj[:,0]
        
        len_s_o1.append(np.linalg.norm(s_atom_coord - o1_atom_coord, axis=-1))
        len_s_o2.append(np.linalg.norm(s_atom_coord - o2_atom_coord, axis=-1))
        
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_s_ar)):
        ax.plot(len_s_o1[i], len_s_o2[i], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    ax.set_xlabel("S-O1 bond length (angstrom)", fontsize=80)
    ax.set_ylabel("S-O2 bond length (angstrom)", fontsize=80)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.set_xlim(1.40, 4.0)
    #ax.set_ylim(1.40, 4.0)
    plt.show()
    plt.close()
    
    
    # plot S-O1 vs S-Ar
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_s_ar)):
        ax.plot(len_s_o1[i], len_s_ar[i], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    ax.set_xlabel("S-O1 bond length (angstrom)", fontsize=80)
    ax.set_ylabel("S-Ar bond length (angstrom)", fontsize=80)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.set_xlim(1.40, 4.0)
    #ax.set_ylim(1.40, 4.0)
    plt.show()
    plt.close()
    
    # plot S-O2 vs S-Biph
    fig, ax = plt.subplots(figsize=(40, 30))
    for i in range(0, len(len_s_ar)):
        ax.plot(len_s_o2[i], len_s_biph[i], '-o', linewidth=0.5, markersize=0.5, color='#1f77b4')
    ax.set_xlabel("S-O2 bond length (angstrom)", fontsize=80)
    ax.set_ylabel("S-Biphenyl bond length (angstrom)", fontsize=80)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.set_xlim(1.40, 4.0)
    #ax.set_ylim(1.40, 4.0)
    plt.show()
    plt.close()
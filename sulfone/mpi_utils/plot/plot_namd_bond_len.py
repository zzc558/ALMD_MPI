#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 00:24:44 2023

@author: chen
"""
import os
import numpy as np
import matplotlib.pyplot as plt

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
    
BOND_LIMIT = {'OS': 2.5, 'CS': 2.5, 'CC': 2.3, 'CH': 6.0}

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

def readXYZs(filename):
    infile=open(filename,"r")
    coords=[[]]
    elements=[[]]
    for line in infile.readlines():
        if len(line.split())==1 and len(coords[-1])!=0:
            coords.append([])
            elements.append([])
        elif len(line.split())==4:
            elements[-1].append(line.split()[0].capitalize())
            coords[-1].append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    return coords,elements

if __name__ == "__main__":
    #fname = "/home/chen/Documents/blueOLED/BlueOledData/src/Traj_from_S6/Traj2/dyn11_TTA6.xyz"
    #coords, elements = readXYZs(fname)
    #coords = np.array(coords, dtype=float)
    #print(coords.shape)
    traj = []
    namd_dir = "/home/chen/Documents/blueOLED/BlueOledData/src"
    for root, dirs, files in os.walk(namd_dir):
        for f in files:
            if f.endswith(".xyz"):
                coords, _ = readXYZs(os.path.join(root, f))
                try:
                    coords = np.array(coords, dtype=float)[:2000]
                except:
                    continue
                traj.append(coords)
            
    traj = np.array(traj, dtype=float)
    
    c1_atom_coord = traj[9:10][:,:,3]
    c2_atom_coord = traj[9:10][:,:,13]
    s_atom_coord = traj[9:10][:,:,0]
    len_s_ar = np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1)
    len_s_biph = np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1)
    
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(len_s_ar, len_s_biph, color='cyan', label='Init dataset')
    #ax.scatter(len_s_ar_init, len_s_biph_init, color='red', s=0.5, label='Init dataset')
    ax.set_title("Sulfur Carbon Length")
    ax.set_xlabel("S-Ar Bond Length")
    ax.set_ylabel("S-Biphenyl Bond Length")
    ax.set_xlim(1.0, 4.0)
    ax.set_ylim(1.0, 4.0)
    #ax.legend()
    plt.show()
    plt.close()
    
    n_traj_break = 0
    for i in range(0, traj.shape[0]):
        for step in traj[i]:
            if not check_bond_length(step)[0]:
                print(i)
                n_traj_break += 1
                break
            
    print(n_traj_break)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:47:25 2023

@author: chen
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np
import os, pickle

def exportXYZs(coords,elements,filename):
    outfile=open(filename,"w")
    for idx in range(len(coords)):
        outfile.write("%i\n\n"%(len(elements[idx])))
        for atomidx,atom in enumerate(coords[idx]):
            outfile.write("%s %f %f %f\n"%(elements[idx][atomidx].capitalize(),atom[0],atom[1],atom[2]))
    outfile.close()

def find_H():
    #al_data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/traj_data_1_1000.npy"
    #init_data_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/results/eg_model_n5000_e1000/eg"
    
    #with open(al_data_path, 'rb') as fh:
    #    coords = np.load(fh)[1:]
    #    engs = np.load(fh)[1:]
    #    states = np.load(fh)[1:]
    
    #print(coords.shape)
    #print(engs.shape)
    #print(states.shape)
    elements = ["S", "O", "O", 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H']
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    len_c_h1 = []
    len_c_h2 = []
    coords = np.empty((1, 200, 38, 3), dtype=float)
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            with open(os.path.join(data_dir, f), 'rb') as fh:
                coords_tmp = np.load(fh)[1:]
            c_atom_coord = coords_tmp[:,:,34]
            h1_atom_coord = coords_tmp[:,:,35]
            h2_atom_coord = coords_tmp[:,:,36]
            coords = np.append(coords, coords_tmp, axis=0)

            len_c_h1.append(np.linalg.norm(c_atom_coord - h1_atom_coord, axis=-1))
            len_c_h2.append(np.linalg.norm(c_atom_coord - h2_atom_coord, axis=-1))
            break
        
    coords = coords[1:]
    len_c_h1 = np.array(len_c_h1, dtype=float)[0]
    len_c_h2 = np.array(len_c_h2, dtype=float)[0]
    
    print(len_c_h1.shape)
    print(np.where(len_c_h1>3))
    print(coords[1252].shape)
    
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(len_c_h1[1252], len_c_h2[1252], '-o', linewidth=0.5, markersize=0.5, color='cyan')
    plt.show()
    plt.close()
    
    elements = [elements] * len(coords[1252])
    exportXYZs(coords[1252], elements, 'traj_1252.xyz')
    
def find_s_ar():
    elements = ["S", "O", "O", 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H']
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    
    ar_1 = False
    ar_6 = False
    biph_1 = False
    biph_6 = False
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            with open(os.path.join(data_dir, f), 'rb') as fh:
                coords = np.load(fh)[1:]
                engs = np.load(fh)[1:]
                states = np.load(fh)[1:]
            try:
                c1_atom_coord = coords[:,:,3]
            except:
                continue
            c2_atom_coord = coords[:,:,13]
            s_atom_coord = coords[:,:,0]
            l1 = np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1)
            l2 = np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1)
            l1_idx = list(set(np.where(l1>3.0)[0]))
            l2_idx = list(set(np.where(l2>3.0)[0]))
            c_s_ar = coords[l1_idx]
            c_s_biph = coords[l2_idx]
            s_s_ar = states[l1_idx]
            s_s_biph = states[l2_idx]
            print(c_s_ar.shape)
            print(s_s_ar.shape)
            if not ar_1:
                i_ar = np.where(s_s_ar[:,0] == 1)[0]
                if i_ar.shape[0] > 0:
                    c_s_ar_1 = c_s_ar[i_ar[0]]
                    exportXYZs(c_s_ar_1, [elements,]*c_s_ar_1.shape[0], 's_ar_1_traj.xyz')
                    ar_1 = True
            if not ar_6:
                i_ar = np.where(s_s_ar[:,0] == 6)[0]
                if i_ar.shape[0] > 0:
                    c_s_ar_6 = c_s_ar[i_ar[0]]
                    exportXYZs(c_s_ar_6, [elements,]*c_s_ar_6.shape[0], 's_ar_6_traj.xyz')
                    ar_6 = True
            if not biph_1:
                i_biph = np.where(s_s_biph[:,0] == 1)[0]
                if i_biph.shape[0] > 0:
                    c_s_biph_1 = c_s_biph[i_biph[0]]
                    exportXYZs(c_s_biph_1, [elements,]*c_s_biph_1.shape[0], 's_biph_1_traj.xyz')
                    biph_1 = True
            if not biph_6:
                i_biph = np.where(s_s_biph[:,0] == 6)[0]
                if i_biph.shape[0] > 0:
                    c_s_biph_6 = c_s_biph[i_biph[0]]
                    exportXYZs(c_s_biph_6, [elements,]*c_s_biph_6.shape[0], 's_biph_6_traj.xyz')
                    biph_6 = True
                    
            if ar_1 and ar_6 and biph_1 and biph_6: break

if __name__ == "__main__":
    find_s_ar()
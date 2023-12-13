#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 00:19:41 2023

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

def plot_al():
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    with open(os.path.join(data_dir, "s_ar.npy"), "rb") as fh:
        len_s_ar = np.load(fh)
        eng_s_ar = np.load(fh)
        state_s_ar = np.load(fh).astype(int)
        
    print(f"len_s_ar shape: {len_s_ar.shape}")
    print(f"eng_s_ar shape: {eng_s_ar.shape}")
    print(f"state_s_ar shape: {state_s_ar.shape}")
    
    for i in range(0, 7):
        print(f"Number of initial state {i}: {np.count_nonzero(state_s_ar[:,0]==i)}")
        
    init_state_plot = 1
    
    idx = np.where(state_s_ar[:,0]==init_state_plot)[0]
    len_selec = len_s_ar[idx]
    eng_selec = eng_s_ar[idx]
    state_selec = state_s_ar[idx]
    
    print(len_selec.shape)
    print(eng_selec.shape)
    print(state_selec.shape)
    
    #idx = np.where(len_selec.flatten() != 0)[0]
    #len_non_zero = len_selec.flatten()[idx]
    #eng_non_zero = np.array([eng_selec[i][state_selec[i]] for i in range(0, state_selec.shape[0])], dtype=float)
    #eng_non_zero = eng_non_zero.flatten()[idx]
    
    #len_eng_dict = {}
    #for i in range(0, len_non_zero.shape[0]):
    #    if len_non_zero[i] in len_eng_dict.keys():
    #        len_eng_dict[len_non_zero[i]].append(eng_non_zero[i])
    #    else:
    #        len_eng_dict[len_non_zero[i]] = [eng_non_zero[i],]
            
    #len_avg = list(len_eng_dict.keys())
    #eng_avg = []
    #for l in len_avg:
    #    eng_avg.append(np.mean(len_eng_dict[l]))
    
    len_non_zero = len_selec[0]
    idx = np.where(len_non_zero != 0)[0]
    len_non_zero = len_non_zero[idx]
    eng_non_zero = np.array([eng_selec[0][i][state_selec[0][i]] for i in range(0, state_selec[0].shape[0])], dtype=float)
    eng_non_zero = eng_non_zero[idx]
    
    print(len_non_zero.shape)
    print(eng_non_zero.shape)
    
    fig, ax = plt.subplots(figsize=[10, 10])
    ax.plot(len_non_zero, eng_non_zero)
    ax.set_xlabel("S-Ar Bond Length", fontsize=20)
    ax.set_ylabel("Energy of Occupied State [Hartree]", fontsize=20)
    plt.show()
    plt.close()
    
def plot_init_dataset():
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    with open(os.path.join(data_dir, "s_biph_init.npy"), "rb") as fh:
        len_s_biph = np.load(fh)
        eng_s_biph = np.load(fh)
        state_s_biph = np.load(fh).astype(int)
        
    print(f"len_s_ar shape: {len_s_biph.shape}")
    print(f"eng_s_ar shape: {eng_s_biph.shape}")
    print(f"state_s_ar shape: {state_s_biph.shape}")

    
    eng_selec = np.array([eng_s_biph[i][state_s_biph[i]] for i in range(0, state_s_biph.shape[0])], dtype=float)
    idx = np.where(eng_selec != 0)[0]
    
    fig, ax = plt.subplots(figsize=[10, 10])
    ax.scatter(len_s_biph[idx], eng_selec[idx])
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    plot_al()
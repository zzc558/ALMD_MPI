#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 09 14:37:29 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Path to coordinates data
    path_to_data = "/home/chen//Documents/blueOLED/NNsForMD/sulfone/results/wo_al/traj_coord.npy"
    path_to_state = "/home/chen//Documents/blueOLED/NNsForMD/sulfone/results/wo_al/traj_state.npy"
    
    with open(path_to_data, "rb") as fh:
        coords = np.load(fh)
        
    print(coords.shape)
    
    # calculate bond length
    c1_atom_coord = coords[:,:,3]
    c2_atom_coord = coords[:,:,13]
    s_atom_coord = coords[:,:,0]
    
    len_s_ar = np.linalg.norm(s_atom_coord - c1_atom_coord, axis=-1)
    len_s_biph = np.linalg.norm(s_atom_coord - c2_atom_coord, axis=-1)
    
    print(len_s_ar.shape)
    print(len_s_biph.shape)
    
    # select bond length from S1 and S6
    with open(path_to_state, 'rb') as fh:
        states = np.load(fh)
    i_s1 = np.where(states[:,0]==1)[0]
    i_s6 = np.where(states[:,0]==6)[0]
    
    print(i_s1.shape)
    print(i_s6.shape)
    
    # calculate average bond length
    s1_len_s_ar_avg = np.average(len_s_ar[i_s1], axis=0)
    s1_len_s_biph_avg = np.average(len_s_biph[i_s1], axis=0)
    
    s6_len_s_ar_avg = np.average(len_s_ar[i_s6], axis=0)
    s6_len_s_biph_avg = np.average(len_s_biph[i_s6], axis=0)

    
    # plot average bond length for S-Ar starting from S1
    fig, ax = plt.subplots()
    ax.plot(s1_len_s_ar_avg)
    ax.set_title("Average Bond Length of S-Ar Starting from S1")
    ax.set_xlabel("Step")
    ax.set_ylabel("Bond Length (Angstrom)")
    plt.show()
    plt.close()
    
    # plot average bond length for S-Biphenyl Ring starting from S1
    fig, ax = plt.subplots()
    ax.plot(s1_len_s_biph_avg)
    ax.set_title("Average Bond Length of S-Biphenyl Ring Starting from S1")
    ax.set_xlabel("Step")
    ax.set_ylabel("Bond Length (Angstrom)")
    plt.show()
    plt.close()
    
    # plot average bond length for S-Ar starting from S6
    fig, ax = plt.subplots()
    ax.plot(s6_len_s_ar_avg)
    ax.set_title("Average Bond Length of S-Ar Starting from S6")
    ax.set_xlabel("Step")
    ax.set_ylabel("Bond Length (Angstrom)")
    plt.show()
    plt.close()
    
    # plot average bond length for S-Biphenyl Ring starting from S6
    fig, ax = plt.subplots()
    ax.plot(s6_len_s_biph_avg)
    ax.set_title("Average Bond Length of S-Biphenyl Ring Starting from S6")
    ax.set_xlabel("Step")
    ax.set_ylabel("Bond Length (Angstrom)")
    plt.show()
    plt.close()
    
    # plot bond length distribution of S-Ar vs S-Biphenyl Ring
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.scatter(len_s_ar, len_s_biph, marker='o', s=0.5, c='cyan')
    ax.set_title("Bond Length Distribution of S-Ar vs S-Biphenyl Ring")
    ax.set_xlabel("S-Ar Bond Length (Angstrom)")
    ax.set_ylabel("S-Biphenyl Bond Length (Angstrom)")
    ax.set_xlim(1.0, 4.0)
    ax.set_ylim(1.0, 4.0)
    plt.show()
    plt.close()
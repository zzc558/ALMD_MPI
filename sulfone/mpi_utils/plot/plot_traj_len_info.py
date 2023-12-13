#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 00:29:59 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
import os, pickle

if __name__ == "__main__":
    len_data_file = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/traj_len_data"
    
    with open(len_data_file, 'rb') as fh:
        traj_len, traj_comp, traj_break, traj_std = pickle.load(fh)
        
    i_min = float("inf")
    for t in traj_len:
        if len(t) < i_min: i_min = len(t)
        
    traj_len = np.array([traj_len[i][:i_min] for i in range(0, len(traj_len))], dtype=int)
    traj_comp = np.array([traj_comp[i][:i_min] for i in range(0, len(traj_comp))], dtype=int)
    traj_break = np.array([traj_break[i][:i_min] for i in range(0, len(traj_break))], dtype=int)
    traj_std = np.array([traj_std[i][:i_min] for i in range(0, len(traj_std))], dtype=int)
    
    # plot average trajectory length
    fig = plt.figure(figsize=(15,10))
    plt.plot(np.average(traj_len, axis=0))
    #plt.title("Average Trajectory Length of 89 MD Workers", fontsize=20)
    plt.xlabel("Number of trajectories per generator", fontsize=20)
    plt.ylabel("Event population", fontsize=20)
    plt.show()
    plt.close()
    
    # plot ratio of trajectory termination reason
    traj_term = np.array([np.sum(traj_comp, axis=0), np.sum(traj_break, axis=0), np.sum(traj_std, axis=0)], dtype=int)
    
    fig = plt.figure(figsize=(30,20))
    plt.plot(traj_term[0]/np.sum(traj_term, axis=0), label="Normal termination")
    plt.plot(traj_term[1]/np.sum(traj_term, axis=0), label="Abnormal bond length")
    plt.plot(traj_term[2]/np.sum(traj_term, axis=0), label="Standard deviation above threshold")
    #plt.title("Average Trajectory Length of 89 MD Workers", fontsize=20)
    plt.xlabel("Active learning progression (a.u.)", fontsize=50)
    plt.ylabel("Fraction of termination reasons", fontsize=50)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    leg = plt.legend(loc="center right", fontsize=30)
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)

    plt.show()
    plt.close()
    
    print(f"Maxium population of bond breaking events is: {np.max(traj_term[1]/np.sum(traj_term, axis=0))}")
    print(f"Minimum number of trajectories of a single generator: {i_min}")
    print(f"Number of generators: {len(traj_len)}")
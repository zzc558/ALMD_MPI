#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 23:10:01 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
import os

K = 1.380649e-23
T = 298
hartree_to_J = 4.35974e-18

if __name__ == "__main__":
    state_start = 5
    ################################
    # Read Active Learning Results #
    ################################
    #data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/e13"
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/states"
    
    state_dist = np.zeros((7, 2000), dtype=float)
    traj_count = 0
    
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            # load state data
            with open(os.path.join(data_dir, f), 'rb') as fh:
                states = np.load(fh).astype(int)
            
            # select finished trajectories (with 2000 time steps) 
            try:
                traj_len = np.count_nonzero(states, axis=1)
            except:
                continue
            traj_choose = np.where(traj_len == 2000)[0]
            states_complete = states[traj_choose]
            
            # select trajectories with specified starting state
            traj_choose = np.where(states_complete[:,0] == state_start)[0]
            states_choose = states_complete[traj_choose]
            
            # calculate state distribution
            for i in range(0, 7):
                i_dist = np.sum(np.where(states_choose == i, 1, 0), axis=0) / states_choose.shape[0]
                state_dist[i] = (state_dist[i] * traj_count + i_dist * traj_choose.shape[0]) / (traj_count + traj_choose.shape[0])
            
            traj_count += traj_choose.shape[0]
    
    print(f"Number of ALMD trajectories is {traj_count}")
    
    #####################################
    # Read Turbomole Calculated Results #
    #####################################
    src_dir = "/home/chen/Documents/blueOLED/BlueOledData/src"

    states = []
    for root, dirs, files in os.walk(os.path.join(src_dir, f'Traj_from_S{state_start}')):
        for f in files:
            if f.endswith('.out'):
                traj = []
                time_curent = 0.0
                with open(os.path.join(root, f), 'r') as fh:
                    content = fh.readlines()
                
                for l in content:
                    if "Molecular dynamics on state" in l:
                        time = float(l.split()[-2])
                        if time < time_curent:
                            traj[-1] = int(l.split()[6]) - 1
                        else:
                            traj.append(int(l.split()[6]) - 1)
                            time_curent += 0.5  
                if len(traj) != 2001: continue
                states.append(traj)
                    
    states = np.array(states, dtype=int)
    state_dist_src = np.zeros((7, 2001), dtype=float)
    
    for i in range(0, 7):
        state_dist_src[i] = np.sum(np.where(states == i, 1, 0), axis=0) / states.shape[0]
    
    #################################
    # Plot AL Results vs QC Results #
    #################################
    fig, ax = plt.subplots(figsize=(30,20))
    for i in range(0, 7):
        ax.plot(state_dist[i], label=f'NN, S{i}', color=f'C{i}', linestyle='-', linewidth=5)
        #ax.plot(state_dist_src[i], label=f'QC, S{i}', color=f'C{i}', linestyle='--')

    leg = ax.legend(loc='center right', fontsize=40, bbox_to_anchor=(1.2, 0.6))
    ax.set_ylabel("State population", fontsize=80)
    ax.set_xlabel("Time Step (0.5 fs)", fontsize=80)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.set_title("Ratio of Occupied State Over Trajectory", fontsize=20)
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(figsize=(30,20))
    for i in range(0, 7):
        #ax.plot(state_dist[i], label=f'NN, S{i}', color=f'C{i}', linestyle='-')
        ax.plot(state_dist_src[i], label=f'QC, S{i}', color=f'C{i}', linestyle='-', linewidth=5)

    leg = ax.legend(loc='center right', fontsize=40, bbox_to_anchor=(1.2, 0.6))
    ax.set_ylabel("State population", fontsize=80)
    ax.set_xlabel("Time Step (0.5 fs)", fontsize=80)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    #ax.set_title("Ratio of Occupied State Over Trajectory", fontsize=20)
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)
    plt.show()
    plt.close()
    
    print(f"Average state distribution of last 250 ALMD trajectories: {[np.mean(state_dist[i][250:]) for i in range(0, 7)]}")
    print(f"Average state distribution of last 250 NAMD trajectories: {[np.mean(state_dist_src[i][250:]) for i in range(0, 7)]}")
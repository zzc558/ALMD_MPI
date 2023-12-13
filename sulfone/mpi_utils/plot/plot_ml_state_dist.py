#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:33:19 2023

@author: chen
"""
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/ml_traj/ml_5ps_nvt/state"
    #data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/ml_traj/ml_5ps/state"
    
    # count of trajectories with length: [<2000, 2000-3999, 4000-5999, 6000-7999, 8000-9999, 10000]
    traj_len_count = [0, 0, 0, 0, 0, 0]
    
    # count trajectories with 10000 steps
    traj_complete = 0
    
    # count total trajectories
    traj_total = 0
    
    # initial state
    state_start = 5
    state_dist = np.zeros((7, 2000), dtype=float)
    traj_count = 0
    
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            # load state data
            with open(os.path.join(data_dir, f), 'rb') as fh:
                states = np.load(fh).astype(int)
                
            traj_total += states.shape[0]
             
            # select finished trajectories (with 2000 time steps) 
            try:
                traj_len = np.count_nonzero(states, axis=1)
            except:
                continue
            
            traj_complete += np.count_nonzero(traj_len == 10000)
            traj_choose = np.where(traj_len >= 2000)[0]
            states_complete = states[traj_choose][:,:2000]
            
            # select trajectories with specified starting state
            traj_choose = np.where(states_complete[:,0] == state_start)[0]
            states_choose = states_complete[traj_choose]
            #print(states_choose.shape)
            
            # calculate state distribution
            for i in range(0, 7):
                i_dist = np.sum(np.where(states_choose == i, 1, 0), axis=0) / states_choose.shape[0]
                state_dist[i] = (state_dist[i] * traj_count + i_dist * traj_choose.shape[0]) / (traj_count + traj_choose.shape[0])
            
            traj_count += traj_choose.shape[0]
    
    print(traj_complete)
    print(traj_total)
    
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
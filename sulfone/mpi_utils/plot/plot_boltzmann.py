#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 01:40:17 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
import os

K = 1.380649e-23
T = 298
hartree_to_J = 4.35974e-18

if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/retrain/process_data"
    state_start = 6
    
    imax = float('inf')
    eng_avg = np.zeros((200, 7), dtype=float)
    state_dist = np.zeros((7, 200), dtype=float)
    traj_count = 0
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            # load energy and state data
            with open(os.path.join(data_dir, f), 'rb') as fh:
                _ = np.load(fh)[1:]
                engs = np.load(fh)[1:]
                states = np.load(fh)[1:].astype(int)
            imax = engs.shape[0] if imax > engs.shape[0] else imax
            
            # select finished trajectories (with 2000 time steps) 
            try:
                traj_len = np.count_nonzero(engs[:,:,0], axis=1)
            except:
                continue
            traj_choose = np.where(traj_len == 200)[0]
            engs_complete = engs[traj_choose]
            states_complete = states[traj_choose]
            #print(engs_complete.shape)
            #print(states_complete.shape)
            
            # select trajectories with specified starting state
            traj_choose = np.where(states_complete[:,0] == state_start)[0]
            states_choose = states_complete[traj_choose]
            engs_choose = engs_complete[traj_choose]
            #print(engs_choose.shape)
            
            # calculate state distribution
            for i in range(0, 7):
                i_dist = np.sum(np.where(states_choose == i, 1, 0), axis=0) / states_choose.shape[0]
                state_dist[i] = (state_dist[i] * traj_count + i_dist * traj_choose.shape[0]) / (traj_count + traj_choose.shape[0])
                
            # calculate average energy
            eng_avg = (np.average(engs_choose, axis=0) * traj_choose.shape[0] + eng_avg * traj_count) / (traj_count + traj_choose.shape[0])
            
            traj_count += traj_choose.shape[0]
    
    print(eng_avg[-1])
    eng_avg = eng_avg.T
    prob_ratio = np.zeros((6, 200), dtype=float)
    for i in range(1, 7):
        prob_ratio[i-1] = np.exp(-((eng_avg[i] - eng_avg[0])*hartree_to_J)/(K*T))
    boltz_dist = prob_ratio / np.sum(prob_ratio, axis=0, keepdims=True)  
    
    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(0, 7):
        ax.plot(state_dist[i], label=f'State {i}', color=f'C{i}')
    #for i in range(0, 6):
    #    ax.plot(boltz_dist[i], linestyle=':', color=f'C{i+1}')
    ax.legend(loc='upper right')
    ax.set_ylabel("Ratio", fontsize=20)
    ax.set_xlabel("Time Step", fontsize=20)
    ax.set_title("Ratio of Occupied State Over Trajectory", fontsize=20)
    plt.show()
    plt.close()

        
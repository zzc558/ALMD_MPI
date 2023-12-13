#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:22:28 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
import os, pickle

if __name__ == "__main__":
    data_dir = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/traj_data/init/process_data"
    traj_length = []
    imax = float('inf')
    for f in os.listdir(data_dir):
        if f.endswith(".npy"):
            with open(os.path.join(data_dir, f), 'rb') as fh:
                coord = np.load(fh)[1:]
            try:
                traj_length.append(np.count_nonzero(coord[:,:,0,0], axis=1))
                imax = coord.shape[0] if imax > coord.shape[0] else imax
            except:
                continue
            
         
    avg_length = []
    for i in range(0, imax):
        tmp = []
        for l in traj_length:
            if len(l) > i:
                tmp.append(l[i])
        avg_length.append(np.average(tmp))
    
    # plot average trajectory length
    fig = plt.figure(figsize=(15,10))
    plt.plot(avg_length)
    plt.title("Average Trajectory Length of 89 MD Workers", fontsize=20)
    plt.xlabel("Number of Trajectories per Worker", fontsize=20)
    plt.ylabel("Average Trajectory Length", fontsize=20)
    plt.show()
    plt.close()
    
    test_res_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/testset_res/init/testset_results"
    r2_list = []
    std_list = []
    with open(test_res_path, 'rb') as fh:
        while True:
            try:
                mae, r2, std = pickle.load(fh)
                r2_list.append(r2)
                std_list.append(std)
            except:
                break
    
    r2_list = np.array(r2_list, dtype=float)
    std_list = np.array(std_list, dtype=float)
    
    print(r2_list.shape)
    
    fig = plt.figure(figsize=(15,10))
    for i in range(0, 4):
        plt.plot(r2_list[:,i], label=f"model{i}")
    plt.ylim(0.90, 1.0)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("R2", fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.title("R2 on Test Set After Each Retraining Iteration", fontsize=20)
    plt.show()
    plt.close()
    
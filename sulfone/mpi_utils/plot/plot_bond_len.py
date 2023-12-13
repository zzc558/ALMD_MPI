#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 03:26:51 2023

@author: chen
"""

import matplotlib.pyplot as plt
import numpy as np
import os

BOND_INDEX =  {
               'OS': [[0, 1], [0, 2]], 'CS': [[0, 3], [0, 13]],\
               'CC': [[3, 4], [3, 5], [4, 6], [5, 8], [6, 10], [6, 34],\
                      [8, 10], [13, 14], [13, 15], [14, 16], [15, 18],\
                      [16, 20], [18, 20], [20, 23], [23, 24], [23, 25],\
                      [24, 26], [25, 28], [26, 30], [28, 30]],\
               'CH': [[4, 7], [5, 9], [8, 11], [10, 12], [14, 17], [15, 19],\
                      [16, 21], [18, 22], [24, 27], [25, 29], [26, 31],\
                      [28, 32], [30, 33], [34, 35], [34, 36], [34, 37]]
               }
    
def calc_bond_length(coord):
    c1 = np.expand_dims(coord, axis=0)
    c2 = np.expand_dims(coord, axis=1)
    dist_matrix = np.sqrt(np.sum(np.square(c1-c2), axis=-1))
    bond_length = {}
    for b, idx in BOND_INDEX.items():
        bond_length[b] = [dist_matrix[i,j] for (i,j) in idx]
    return bond_length

def calc_state_distribution(traj_state, states):
    state_distribution = []
    for s in states:
        state_distribution.append(np.sum((traj_state == s).astype(float), axis=0)/traj_state.shape[0])
    return state_distribution

if __name__ == "__main__":
    res_dir = "results/n5000_no_al"
    states = list(range(0, 7))
    
    ############################# 
    #  plot state distribution  #
    #############################
    with open(os.path.join(res_dir, 'traj_state.npy'), 'rb') as fh:
        traj_state = np.load(fh)
    print(f"Shape of trajectories states: {traj_state.shape}")
    
    #### extract trajectories starting from S1 ###
    traj_from_S1 = traj_state[np.nonzero(traj_state[:,0] == 1)[0]]
    print(f"Shape of trajectories states starting from S1: {traj_from_S1.shape}")
    # count number of trajectories end at state s
    for s in states:
        n_s = np.nonzero(traj_from_S1[:,-1] == s)[0].shape[0]
        print(f"Number of states end in S{s}: {n_s}")
    # plot state distribution
    state_distribution_from_S1 = calc_state_distribution(traj_from_S1, states)
    fig = plt.figure()
    for i in range(0, len(state_distribution_from_S1)):
        plt.plot(state_distribution_from_S1[i], label=f"S{i}")
    plt.legend(loc="upper right")
    plt.xlabel("Time Step")
    plt.ylabel("Population ratio")
    plt.title("S1 Trajectories")
    plt.show()
    plt.close()
    
    ### extract trajectories starting from S6 ###
    traj_from_S6 = traj_state[np.nonzero(traj_state[:,0] == 6)[0]]
    print(f"Shape of trajectories states starting from S6: {traj_from_S6.shape}")
    # count number of trajectories end at state s
    for s in states:
        n_s = np.nonzero(traj_from_S6[:,-1] == s)[0].shape[0]
        print(f"Number of states end in S{s}: {n_s}")
    # plot state distribution
    state_distribution_from_S6 = calc_state_distribution(traj_from_S6, states)
    fig = plt.figure()
    for i in range(0, len(state_distribution_from_S6)):
        plt.plot(state_distribution_from_S6[i], label=f"S{i}")
    plt.legend(loc="upper right")
    plt.xlabel("Time Step")
    plt.ylabel("Population ratio")
    plt.title("S6 Trajectories")
    plt.show()
    plt.close()
    
    ############################# 
    #      plot bond length     #
    #############################
    with open(os.path.join(res_dir, 'traj_coord.npy'), 'rb') as fh:
        traj_coord = np.load(fh)
    print(f"Shape of trajectories coordinates: {traj_coord.shape}")
    
    ### extract trajectories starting from S1 ###
    traj_coord_from_S1 = traj_coord[np.nonzero(traj_state[:,0] == 1)[0]]
    print(f"Shape of trajectories coordinates starting from S1: {traj_coord_from_S1.shape}")
    # calculate interested bond length
    cs_bond_1 = np.empty((traj_coord_from_S1.shape[0], traj_coord_from_S1.shape[1]), dtype=float)
    cs_bond_2 = np.empty((traj_coord_from_S1.shape[0], traj_coord_from_S1.shape[1]), dtype=float)
    cc_bond_14 = np.empty((traj_coord_from_S1.shape[0], traj_coord_from_S1.shape[1]), dtype=float)
    cc_bond_4 = np.empty((traj_coord_from_S1.shape[0], traj_coord_from_S1.shape[1]), dtype=float)
    for i in range(0, len(traj_coord_from_S1)):
        for j in range(0, len(traj_coord_from_S1[i])):
            bond_length = calc_bond_length(traj_coord_from_S1[i][j])
            cs_bond_1[i][j] = bond_length['CS'][0]
            cs_bond_2[i][j] = bond_length['CS'][1]
            cc_bond_14[i][j] = bond_length['CC'][BOND_INDEX['CC'].index([20, 23])]
            cc_bond_4[i][j] = bond_length['CC'][BOND_INDEX['CC'].index([6, 34])]
    # plot average bond length
    cs_bond_1_mean = np.mean(cs_bond_1, axis=0)
    fig = plt.figure()
    plt.plot(cs_bond_1_mean, label='Dist. b/w atom 1 & 4')
    plt.legend(loc='upper right')
    plt.xlabel('Time Step')
    plt.ylabel('Avg. Bond Length (Euclidean dist.)')
    plt.title("S1 Trajectories C-S (Bond 1) Length")
    plt.show()
    plt.close()
    
    cs_bond_2_mean = np.mean(cs_bond_2, axis=0)
    fig = plt.figure()
    plt.plot(cs_bond_2_mean, label='Dist. b/w atom 1 & 14')
    plt.legend(loc='upper right')
    plt.xlabel('Time Step')
    plt.ylabel('Avg. Bond Length (Euclidean dist.)')
    plt.title("S1 Trajectories C-S (Bond 2) Length")
    plt.show()
    plt.close()
    
    cc_bond_14_mean = np.mean(cc_bond_14, axis=0)
    fig = plt.figure()
    plt.plot(cc_bond_14_mean, label='Dist. b/w atom 21 & 24')
    plt.legend(loc='upper right')
    plt.xlabel('Time Step')
    plt.ylabel('Avg. Bond Length (Euclidean dist.)')
    plt.title("S1 Trajectories C-C (Bond 14) Length")
    plt.show()
    plt.close()
    
    cc_bond_4_mean = np.mean(cc_bond_4, axis=0)
    fig = plt.figure()
    plt.plot(cc_bond_4_mean, label='Dist. b/w atom 6 & 34')
    plt.legend(loc='upper right')
    plt.xlabel('Time Step')
    plt.ylabel('Avg. Bond Length (Euclidean dist.)')
    plt.title("S1 Trajectories C-C (Bond 4) Length")
    plt.show()
    plt.close()
    
    ### extract trajectories starting from S6 ###
    traj_coord_from_S6 = traj_coord[np.nonzero(traj_state[:,0] == 6)[0]]
    print(f"Shape of trajectories coordinates starting from S1: {traj_coord_from_S1.shape}")
    # calculate interested bond length
    cs_bond_1 = np.empty((traj_coord_from_S6.shape[0], traj_coord_from_S6.shape[1]), dtype=float)
    cs_bond_2 = np.empty((traj_coord_from_S6.shape[0], traj_coord_from_S6.shape[1]), dtype=float)
    cc_bond_14 = np.empty((traj_coord_from_S6.shape[0], traj_coord_from_S6.shape[1]), dtype=float)
    cc_bond_4 = np.empty((traj_coord_from_S6.shape[0], traj_coord_from_S6.shape[1]), dtype=float)
    for i in range(0, len(traj_coord_from_S6)):
        for j in range(0, len(traj_coord_from_S6[i])):
            bond_length = calc_bond_length(traj_coord_from_S6[i][j])
            cs_bond_1[i][j] = bond_length['CS'][0]
            cs_bond_2[i][j] = bond_length['CS'][1]
            cc_bond_14[i][j] = bond_length['CC'][BOND_INDEX['CC'].index([20, 23])]
            cc_bond_4[i][j] = bond_length['CC'][BOND_INDEX['CC'].index([6, 34])]
    # plot average bond length
    cs_bond_1_mean = np.mean(cs_bond_1, axis=0)
    fig = plt.figure()
    plt.plot(cs_bond_1_mean, label='Dist. b/w atom 1 & 4')
    plt.legend(loc='upper right')
    plt.xlabel('Time Step')
    plt.ylabel('Avg. Bond Length (Euclidean dist.)')
    plt.title("S6 Trajectories C-S (Bond 1) Length")
    plt.show()
    plt.close()
    
    cs_bond_2_mean = np.mean(cs_bond_2, axis=0)
    fig = plt.figure()
    plt.plot(cs_bond_2_mean, label='Dist. b/w atom 1 & 14')
    plt.legend(loc='upper right')
    plt.xlabel('Time Step')
    plt.ylabel('Avg. Bond Length (Euclidean dist.)')
    plt.title("S6 Trajectories C-S (Bond 2) Length")
    plt.show()
    plt.close()
    
    cc_bond_14_mean = np.mean(cc_bond_14, axis=0)
    fig = plt.figure()
    plt.plot(cc_bond_14_mean, label='Dist. b/w atom 21 & 24')
    plt.legend(loc='upper right')
    plt.xlabel('Time Step')
    plt.ylabel('Avg. Bond Length (Euclidean dist.)')
    plt.title("S6 Trajectories C-C (Bond 14) Length")
    plt.show()
    plt.close()
    
    cc_bond_4_mean = np.mean(cc_bond_4, axis=0)
    fig = plt.figure()
    plt.plot(cc_bond_4_mean, label='Dist. b/w atom 6 & 34')
    plt.legend(loc='upper right')
    plt.xlabel('Time Step')
    plt.ylabel('Avg. Bond Length (Euclidean dist.)')
    plt.title("S6 Trajectories C-C (Bond 4) Length")
    plt.show()
    plt.close()
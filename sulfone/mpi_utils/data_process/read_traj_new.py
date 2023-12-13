#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 03:26:45 2023

@author: chen
"""

import numpy as np
import os, pickle


def read_traj_data(traj_data, target: str):
    state_all = []
    energy_all = []

    # extract data of trajectories that are already completed
    i_complete_end = np.where(traj_data == -3)[0][0]
    data_complete = traj_data[:i_complete_end]
    


def read_state_complete(data_complete, data_seperator):
    # different data are seperated by data_seperator
    # state is the fourth part
    # remove the last -1 to avoid empty array from np.split()
    state_data = data_complete[data_seperator[2]+1:data_seperator[3]-1]
    # different trajectories are seperated by -1
    state = [s[s!=-1] for s in np.split(state_data, np.where(state_data == -1)[0], axis=0)]
    return state

def read_state_incomplete(data_incomplete, data_seperator):
    # different data are seperated by data_seperator
    # state is the fourth part
    state = data_incomplete[data_seperator[2]+1:data_seperator[3]]
    state = state[state!=-1]
    return state

def read_energy_complete(data_complete, data_seperator):
    # different data are seperated by data_seperator
    # energy is the second part
    # remove the last -1 to avoid empty array from np.split()
    energy_data = data_complete[data_seperator[0]+1:data_seperator[1]-1]
    # different trajectories are seperated by -1
    energy = [e[e!=-1] for e in np.split(energy_data, np.where(energy_data == -1)[0], axis=0)]
    energy = [e.reshape(int(e.shape[0]/n_state), n_state) for e in energy]
    return energy

def read_energy_incomplete(data_incomplete, data_seperator):
    # different data are seperated by data_seperator
    # energy is the second part
    energy = data_incomplete[data_seperator[0]+1:data_seperator[1]]
    energy = energy[energy!=-1]
    energy = energy.reshape(int(energy.shape[0]/n_state), n_state)
    return energy

if __name__ == "__main__":
    data_dir = "./results/AlRun_1ns_nvt_newscript_run1/molecular_dynamic/trajData"
    #res_dir = "./results/AlRun_1ns_nvt_newscript_run1/molecular_dynamic/trajData/read"
    #os.makedirs(res_dir, exist_ok=True)

    data_target = "coordinate, termination, energy and state"
    record_incomplete = False

    # trajectory parameters
    n_state = 7
    n_atom = 38
    n_steps = 2e6
    step_keys = ['coord', 'energy', 'force', 'state']
    traj_keys = ['termin',]

    # data to read
    state_all = []
    energy_all = []
    for f in os.listdir(data_dir):
        if not f.endswith(".mpi"):
            continue
        
        state_incomplete = np.zeros((1,), dtype=int)
        state_complete = []
        energy_incomplete = np.zeros((1, n_state), dtype=float)
        energy_complete = []
        with open(os.path.join(data_dir, f), 'rb') as fh:
            while True:
                try:
                    traj_data = np.load(fh)
                except:
                    print("Data Loading finished.")
                    break
                
                # extract data of trajectories that are already completed
                i_complete_end = np.where(traj_data == -3)[0][0]
                data_complete = traj_data[:i_complete_end]
                # different type of data are seperated by -2
                data_seperator = np.where(traj_data == -2)[0]
                if "state" in data_target and not (data_complete <= -1).all():
                    state = read_state_complete(data_complete, data_seperator)
                    if state_incomplete.shape[0] > 1:
                        # there is an unfinished trajectory in last saving iteration
                        state[0] = np.append(state_incomplete[1:], state[0], axis=0)
                        state_incomplete = np.zeros((1,), dtype=int)
                    state_all += state
                if "energy" in data_target and not (data_complete <= -1).all():
                    energy = read_energy_complete(data_complete, data_seperator)
                    if energy_incomplete.shape[0] > 1:
                        # there is an unfinished trajectory in last saving iteration
                        energy[0] = np.append(energy_incomplete[1:], energy[0], axis=0)
                        energy_incomplete = np.zeros((1, n_state), dtype=float)
                    energy_all += energy

                # extract data of trajectories that are incomplete
                i_incomplete_begin = np.where(traj_data == -4)[0][0] + 1
                i_incomplete_end = np.where(traj_data == -5)[0][0]
                data_incomplete = traj_data[i_incomplete_begin:i_incomplete_end]
                # different type of data are seperated by -2
                data_seperator = np.where(data_incomplete == -2)[0]
                if "state" in data_target and not (data_incomplete <= -1).all():
                    state = read_state_incomplete(data_incomplete, data_seperator)
                    state_incomplete = np.append(state_incomplete, state, axis=0)
                if "energy" in data_target and not (data_incomplete <= -1).all():
                    energy = read_energy_incomplete(data_incomplete, data_seperator)
                    energy_incomplete = np.append(energy_incomplete, energy, axis=0)
        
        if record_incomplete:
            if "state" in data_target:
                state_all.append(state_incomplete[1:])
            if "energy" in data_target:
                energy_all.append(energy_incomplete[1:])

    print(f"Number of trajectories recorded for state data: {len(state_all)}")
    print(f"Number of trajectories recorded for energy data: {len(energy_all)}")

    if "state" in data_target:
        with open(os.path.join(data_dir, "state_all.pickle"), "wb") as fh:
            pickle.dump(state_all, fh)
    if "energy" in data_target:
        with open(os.path.join(data_dir, "energy_all.pickle"), "wb") as fh:
            pickle.dump(energy_all, fh)
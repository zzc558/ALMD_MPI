#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:10:00 2023

@author: chen
"""

import numpy as np
import os, pickle
    

if __name__ == "__main__":
    data_dir = "./results/AlRun_1ns_nvt_newscript_run1/molecular_dynamic/trajData"
    termin_dir = "./results/AlRun_1ns_nvt_newscript_run1/molecular_dynamic/trajData/terminate"
    step_keys = ['coord', 'energy', 'force', 'state']    # list of keys of traj_data to data related to each time step
    n_atom = 38
    termin_code = 1 # 0 for normal termination, 1 for bond breaking, 4 for high STD
    if termin_code == 1:
        save_file = "bond_breaking_traj.pickle"
    elif termin_code == 4:
        save_file = "high_std_traj.pickle"
    else:
        save_file = "normal_traj.pickle"
    
    coords_record = []
    for f in os.listdir(termin_dir):
        if not f.endswith('.mpi'):
            continue
        
        # read trajectory termination reasons
        with open(os.path.join(termin_dir, f), 'rb') as fh:
            termin_reasons = np.load(fh)
        if len(termin_reasons) == 0 or not (termin_reasons == termin_code).any():
            # skip trajectories terminated normally or with high STD events
            continue
        
        # load data of trajectories terminated with bond breaking event
        coord_incomplete = []
        coord_complete = []
        
        with open(os.path.join(data_dir, f), 'rb') as fh:
            while True:
                try:
                    traj_data = np.load(fh)
                except:
                    break
                
                # extract data of trajectories that are incomplete
                i_incomplete_begin = np.where(traj_data == -4)[0][0] + 1
                i_incomplete_end = np.where(traj_data == -5)[0][0]
                data_incomplete = traj_data[i_incomplete_begin:i_incomplete_end]
                if not (data_incomplete <= -1).all():
                    # extract the first part of the data, which are coordinates
                    i_coord_end = np.where(data_incomplete == -2)[0][0]
                    coord = data_incomplete[:i_coord_end]
                    coord = coord[coord!=-1]
                    coord_incomplete = np.append(coord_incomplete, coord, axis=0)
                
                # extract data of trajectories that are already completed
                i_complete_end = np.where(traj_data == -3)[0][0]
                data_complete = traj_data[:i_complete_end]
                if not (data_complete <= -1).all():
                    # extract the first part of the data, which are coordinates
                    i_coord_end = np.where(data_complete == -1)[0][0]
                    coord = data_complete[:i_coord_end]
                    if len(coord_incomplete) > 0:
                        coord = np.append(coord_incomplete, coord, axis=0)
                        coord_complete.append(coord)
                        coord_incomplete = []
        
        print(termin_reasons)
        print(len(coord_complete))
        
        for i in range(0, len(termin_reasons)):
            if termin_reasons[i] == termin_code:
                coords_record.append(coord_complete[i].reshape(int(coord_complete[i].shape[0]/(n_atom*3)), n_atom, 3))
                
    with open(os.path.join(data_dir, save_file), 'wb') as fh:
        pickle.dump(coords_record, fh)
                
                    
            
        
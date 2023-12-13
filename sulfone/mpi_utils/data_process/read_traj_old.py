#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:05:15 2023

@author: chen

Function:
    Read trajectory data (coordinates, energy and force) and save the processed data.
"""

import numpy as np
import os

if __name__ == "__main__":
    data_dir = "../results/MLRun_5ps_nvt_complete/molecular_dynamic/trajData/"
    
    coords = []
    for f in os.listdir(data_dir):
        if not f.endswith('.npy'):
            continue
        coords.append([])
        with open(os.path.join(data_dir, f), 'rb') as fh:
            try:
                while True:
                    data = np.load(fh)
                    i = np.where(data==-1)[0]
                    coords[-1] += data[:i[0]].reshape(int((i[0]+1)/(2000*38*3)), 2000, 38, 3).tolist()
            except:
                continue
    coords = np.array(coords, dtype=float)
    
    with open(os.path.join(data_dir, 'traj_coordinates.npy', 'wb')) as fh:
        np.save(fh, coords)
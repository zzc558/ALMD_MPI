#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:25:25 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    src_dir = "/home/chen/Documents/blueOLED/BlueOledData/src"
    state_start = 6
    
    engs = []
    states = []
    for root, dirs, files in os.walk(os.path.join(src_dir, f'Traj_from_S{state_start}')):
        for f in files:
            if f.endswith('.dat'):
                traj = []
                with open(os.path.join(root, f), 'r') as fh:
                    content = fh.readlines()
                
                for l in content:
                    traj.append(l.split()[1:-2])
                if len(traj) != 2001: continue
                engs.append(traj)
            
            elif f.endswith('.out'):
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
                states.append(traj)
                    
    engs = np.array(engs, dtype=float)
    states = np.array(states, dtype=int)
    state_dist = np.zeros((7, 2001), dtype=float)
    
    for i in range(0, 7):
        state_dist[i] = np.sum(np.where(states == i, 1, 0), axis=0) / states.shape[0]
        
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(0, 7):
        ax.plot(state_dist[i], label=f'State {i}', color=f'C{i}')
    ax.legend(loc='upper left')
    ax.set_ylabel("Ratio", fontsize=20)
    ax.set_xlabel("Time Step", fontsize=20)
    ax.set_title("Ratio of Occupied State Over Trajectory", fontsize=20)
    plt.show()
    plt.close()

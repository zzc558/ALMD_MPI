#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:01:11 2023

@author: chen
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    res_dir = '../results/TestRun_n5000_01'
    
    with open(os.path.join(res_dir, 'passive_learner', 'pltime.json'), 'r') as fh:
        pltime = json.load(fh)
    for k, v in pltime.items():
        print(f"PL {k} mean time: {np.mean(np.array(v, dtype=float))}")
        print(f"PL {k} min time: {np.min(np.array(v, dtype=float))}")
        print(f"PL {k} max time: {np.max(np.array(v, dtype=float))}")
        print()
        plt.plot(np.array(v, dtype=float))
        plt.show()
    
    md_gather_time = []
    md_scatter_time = []
    md_prop_time = []
    for file in os.listdir(os.path.join(res_dir, 'molecular_dynamic')):
        if 'mdtime' in file:
            with open(os.path.join(res_dir, 'molecular_dynamic', file), 'r') as fh:
                md_time = json.load(fh)
            md_gather_time += md_time['gather']
            md_scatter_time += md_time['scatter']
            md_prop_time += md_time['prop']
    print(f"MD propagation time: {np.mean(np.array(md_prop_time, dtype=float))}")
    print(f"MD gather time: {np.median(np.array(md_gather_time, dtype=float))}")
    print(f"MD gather min time: {np.min(np.array(md_gather_time, dtype=float))}")
    print(f"MD gather max time: {np.max(np.array(md_gather_time, dtype=float))}")
    print(f"MD scatter time: {np.mean(np.array(md_scatter_time, dtype=float))}")
    
    
    print()
    """
    #n_steps = []
    #for file in os.listdir(os.path.join(res_dir, 'molecular_dynamic')):
    #    if 'traj_data' in file:
    #        try:
    #            n_steps.append([])
    #            with open(os.path.join(res_dir, 'molecular_dynamic', file), 'r') as fh:
    #                res = json.load(fh)
    #            for i in range(0, 800):
    #                n_steps[-1].append(len(res['energy'][i]))
    #        except:
    #            continue
            
    #n_steps = np.array(n_steps, dtype=int)
    
    #plt.plot(np.average(n_steps, axis=0))
    
    with open(os.path.join(res_dir, 'machine_learning', 'mltime.json'), 'r') as fh:
        mltime = json.load(fh)
    for k, v in mltime.items():
        print(f"ML {k} time: {np.mean(np.array(v, dtype=float))}")
        
    
    with open(os.path.join(res_dir, "process_manager", "testset_results.json"), 'r') as fh:
        testset_eng_mae, testset_eng_r2, testset_std = json.load(fh)
    
    testset_std = np.array(testset_std, dtype=float)
    std_mean = np.mean(testset_std, axis=1)
    #print(std_mean.shape)
    
    fig = plt.figure()
    for i in range(0, 7):
        plt.plot(std_mean[:,i], label=f'E{i}')
    #plt.plot(np.array(testset_eng_r2[0], dtype=float))
    plt.xlabel("Retrain step")
    plt.ylabel("Mean STD")
    plt.title("Mean STD of Predictions on Test Set")
    plt.legend(loc='lower right')
    plt.show()
    plt.close()
    
    #fig = plt.figure()
    #for i in range(0, 4):
    #    plt.plot(np.array(testset_eng_r2[i], dtype=float))
    #plt.xlabel("Retrain step")
    #plt.ylabel("R2")
    #plt.title("R2 of Predictions on Test Set")
    #plt.legend(loc='lower right')
    #plt.show()
    #plt.close()
    
    #print(np.array(testset_eng_r2, dtype=float).shape)
    
    with open('results/retrain_history0.json', 'r') as fh:
        hist = json.load(fh)
    fig = plt.figure()
    plt.plot(np.array(hist['val_energy_r2'], dtype=float))
    plt.xlabel("Retrain step")
    plt.ylabel("R2")
    plt.title("R2 of Predictions on Validation Set")
    plt.show()
    plt.close()
    """
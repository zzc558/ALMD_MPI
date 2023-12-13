#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 04:02:17 2023

@author: chen
"""
import json, os, pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    #res_path = "../results/testset_res/init_05_rerun/testset_results.json" # results after 9 hours
    res_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/mpi_utils/results/testset_res/retrain/testset_results_latest"
    n_model = 4
    n_state = 7
    
    test_eng_mae = np.zeros((n_model, 1), dtype=float)
    test_eng_r2 = np.zeros((n_model, 1), dtype=float)
    
    with open(res_path, 'rb') as fh:
        while True:
            try:
                eng_mae, eng_r2, _ = pickle.load(fh)
            except:
                break
            test_eng_mae = np.append(test_eng_mae, eng_mae, axis=1)
            test_eng_r2 = np.append(test_eng_r2, eng_r2, axis=1)

    test_eng_mae = test_eng_mae[:,1:]
    test_eng_r2 = test_eng_r2[:,1:]
    
    print(test_eng_mae.shape)
    print(test_eng_r2.shape)
    
    eng_mean = np.mean(test_eng_mae, axis=0)
    print(eng_mean.shape)
    
    fig, ax = plt.subplots(figsize=(30, 20))
    ax.plot(eng_mean)
    ax.set_xlabel("Number of retrain iteration", fontsize=50)
    ax.set_ylabel("Energy mean absolute error (eV)", fontsize=50)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    plt.show()
    plt.close()
    
    """
    fig = plt.figure()
    for i in range(0, n_model):
        plt.plot(test_eng_r2[0,i::4], label=f"model{i}")
    plt.ylim(0.90, 0.96)
    plt.xlabel("Iteration")
    plt.ylabel("R2")
    plt.legend(loc="lower right")
    plt.title("R2 on Test Set After Each Retraining Iteration")
    plt.show()
    plt.close()
    
    std_mean = np.mean(test_eng_std, axis=1)
    print(std_mean.shape)
    fig = plt.figure(figsize=(25, 15))
    for i in range(0, n_state):
        plt.plot(std_mean[:,i], label=f"E{i}")
    plt.legend(loc="upper right", fontsize=30)
    plt.xlabel("Iteration", fontsize=30)
    plt.ylabel("STD", fontsize=30)
    plt.title("STD on Test Set After Each Retraining Iteration", fontsize=50)
    plt.show()
    plt.close()
    """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:35:09 2023

@author: chen
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    retrain_hist_dir = "../results/retrain_history/init_std_05"
    n_model = 4
    
    model_hist = {}
    for i in range(0, n_model):
        hist_path = os.path.join(retrain_hist_dir, f"retrain_history{i}.json")
        with open(hist_path, 'r') as fh:
            hist = json.load(fh)
        for k, v in hist.items():
            if k in model_hist:
                model_hist[k].append(v)
            else:
                model_hist[k] = [v,]
    
    # plot energy MAE vs epoch
    for i in range(0, n_model):
        fig = plt.figure(figsize=(25, 15))
        plt.plot(np.arange(0, len(model_hist["energy_mean_absolute_error"][i]), dtype=int), np.array(model_hist["energy_mean_absolute_error"][i], dtype=float), label=f'Model{i}_Train')
        plt.plot(np.arange(0, len(model_hist["val_energy_mean_absolute_error"][i])*10, 10, dtype=int), np.array(model_hist["val_energy_mean_absolute_error"][i], dtype=float), label=f'Model{i}_Val')
        #plt.ylim(0.02, 0.1)
        plt.legend(loc="upper right", fontsize=30)
        plt.xlabel("Epoch", fontsize=30)
        plt.ylabel("Mean Absolute Error [eV]", fontsize=30)
        plt.title(f"Mean Absolute Error of Model {i} for Active Learning", fontsize=50)
        plt.show()
        plt.close()
        
    # plot energy r2 vs epoch
    for i in range(0, n_model):
        fig = plt.figure(figsize=(25, 15))
        plt.plot(np.arange(0, len(model_hist["energy_r2"][i]), dtype=int), np.array(model_hist["energy_r2"][i], dtype=float), label=f'Model{i}_Train')
        plt.plot(np.arange(0, len(model_hist["val_energy_r2"][i])*10, 10, dtype=int), np.array(model_hist["val_energy_r2"][i], dtype=float), label=f'Model{i}_Val')
        #plt.ylim(0.99, 1)
        plt.legend(loc="lower right", fontsize=30)
        plt.xlabel("Epoch", fontsize=30)
        plt.ylabel("R2", fontsize=30)
        plt.title(f"R2 of Model {i} for Active Learning", fontsize=50)
        plt.show()
        plt.close()
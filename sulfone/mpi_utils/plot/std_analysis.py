#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 03:46:47 2023

@author: chen
"""

import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

#def mask_mae(y_true, y_pred):
#    mask = np.not_equal(y_true, 0.0).astype(float)
#    return np.sum(np.abs(y_true*mask - y_pred*mask), axis=0)/np.sum(mask, axis=0)

def mask_abs_error(y_true, y_pred):
    mask = np.not_equal(y_true, 0.0).astype(float)
    return np.abs(y_true*mask - y_pred*mask)

if __name__ == "__main__":
    with open("/home/chen/Documents/blueOLED/NNsForMD/sulfone/data/eg_model_n5000_e1000_prediction_results", 'rb') as fh:
        res = pickle.load(fh)
        
    eng_pred = []
    f_pred = []
    for r in res['eg']:
        eng_pred.append(r[0])
        f_pred.append(r[1])
        
    eng_pred = np.array(eng_pred, dtype=float)
    f_pred = np.array(f_pred, dtype=float)
    print(f"energy prediction shape {eng_pred.shape}")
    print(f"force prediction shape {f_pred.shape}")
    
    std = np.std(eng_pred, axis=0, ddof=1)
    std2 = np.std(eng_pred[1:], axis=0, ddof=1)
    print(f"standard deviation shape: {std.shape}")
    print(f"max std: {np.max(std)}")
    print(f"min std: {np.min(std)}")
    print(f"mean std: {np.mean(std)}")
    print(f"mean std2: {np.mean(std2)}")
    
    plt.figure()
    plt.scatter(eng_pred[0,:,0].flatten(), eng_pred[1,:,0].flatten())
    plt.xlabel("model0")
    plt.ylabel("model1")
    plt.show()
    plt.close()
    plt.figure()
    plt.scatter(eng_pred[1,:,0].flatten(), eng_pred[2,:,0].flatten())
    plt.xlabel("model1")
    plt.ylabel("model2")
    plt.show()
    plt.close()

    
    plt.figure()
    plt.scatter(f_pred[0,:,0].flatten(), f_pred[1,:,0].flatten())
    plt.xlabel("f model0")
    plt.ylabel("f model1")
    plt.show()
    plt.close()
    plt.figure()
    plt.scatter(f_pred[1,:,0].flatten(), f_pred[2,:,0].flatten())
    plt.xlabel("f model1")
    plt.ylabel("f model2")
    plt.show()
    plt.close()
    
    
    std_limit = 0.5
    ratio = np.sum((std > std_limit).astype(float))/(std.shape[0]*std.shape[1])
    print(f"ratio of predictions with std greater than {std_limit}: {ratio}")
    
    with open('/home/chen/Documents/blueOLED/NNsForMD/sulfone/data/testing_23604_random_order_correct.json', 'r') as fh:
        _, engs, _ = json.load(fh)
        
    engs = np.array(engs, dtype=float) * 27.21138624598853  # Hatree to eV
    print(f"Ground truth energy shape: {engs.shape}")
    
    #error = np.mean(mask_abs_error(engs, eng_pred), axis=0)
    error = np.mean(np.abs((engs-eng_pred)), axis=0)
    print(f"Mean absolute error shape: {error.shape}")
    
    print(f"Max mae: {np.max(error)}")
    print(f"min mae: {np.min(error)}")
    
    # remove error and std for energy predictions without ground truth
    # plot results according to differnet states
    idx_state = [np.nonzero(engs[:,i]) for i in range(0, 7)]
    std_state = [std[:,i][idx_state[i]] for i in range(0, 7)]
    error_state = [error[:,i][idx_state[i]] for i in range(0, 7)]
    
    # plot mae vs std
    fig, ax = plt.subplots()
    h = ax.hist2d(error.flatten(), std.flatten(), bins=100, cmap='autumn')
    fig.colorbar(h[3], ax=ax)
    ax.set_title("Energy Prediction MAE vs STD")
    ax.set_xlabel("MAE [eV]")
    ax.set_ylabel("STD")
    plt.show()
    plt.close()
    
    # plot mae vs std
    fig, ax = plt.subplots(figsize=(30, 20))
    h = ax.scatter(error.flatten(), std.flatten())
    ax.set_title("Energy Prediction MAE vs STD", fontsize=20)
    ax.set_xlabel("MAE [eV]", fontsize=20)
    ax.set_ylabel("STD [eV]", fontsize=20)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 5)
    plt.show()
    plt.close()
    
    # plot mae vs std
    fig, ax = plt.subplots(figsize=(30, 20))
    for i in range(0, 7):
        ax.scatter(error_state[i], std_state[i], label=f"S{i}")
    #ax.set_title("Energy Prediction MAE vs STD", fontsize=20)
    ax.set_xlabel("Mean absolute error (eV)", fontsize=40)
    ax.set_ylabel("Standard deviation (eV)", fontsize=40)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.set_xlim(0, 0.00125)
    #ax.set_ylim(0.0, 4.5)
    ax.legend(fontsize=15, markerscale=2)
    plt.show()
    plt.close()
    
    for i in range(len(error)):
        if np.abs(error[i][1]-0.1)<0.01:
            if np.abs(std[i][1]-0.05)<0.1:
                #print(i, eng_pred[:,i,1]-engs[i][1])
                print(i, eng_pred[:,i,1]-np.mean(eng_pred[:,i,1]))
    	
    """
    # plot mae vs std
    fig, ax = plt.subplots()
    h = ax.scatter(error[:,0].flatten(), std[:,0].flatten())
    ax.set_title("E0 Prediction MAE vs STD")
    ax.set_xlabel("MAE [eV]")
    ax.set_ylabel("STD [eV]")
    ax.set_xlim(0, 0.00125)
    ax.set_ylim(0.0, 4.5)
    plt.show()
    plt.close()
    
    # plot mae vs std
    fig, ax = plt.subplots()
    h = ax.scatter(error[:,1].flatten(), std[:,1].flatten())
    ax.set_title("E1 Prediction MAE vs STD")
    ax.set_xlabel("MAE [eV]")
    ax.set_ylabel("STD [eV]")
    ax.set_xlim(0, 0.00125)
    ax.set_ylim(0.0, 4.5)
    plt.show()
    plt.close()
    
    # plot mae vs std
    fig, ax = plt.subplots()
    h = ax.scatter(error[:,2].flatten(), std[:,2].flatten())
    ax.set_title("E2 Prediction MAE vs STD")
    ax.set_xlabel("MAE [eV]")
    ax.set_ylabel("STD [eV]")
    ax.set_xlim(0, 0.00125)
    ax.set_ylim(0.0, 4.5)
    plt.show()
    plt.close()
    
    # plot mae vs std
    fig, ax = plt.subplots()
    h = ax.scatter(error[:,3].flatten(), std[:,3].flatten())
    ax.set_title("E3 Prediction MAE vs STD")
    ax.set_xlabel("MAE [eV]")
    ax.set_ylabel("STD [eV]")
    ax.set_xlim(0, 0.00125)
    ax.set_ylim(0.0, 4.5)
    plt.show()
    plt.close()
    
    # plot mae vs std
    fig, ax = plt.subplots()
    h = ax.scatter(error[:,4].flatten(), std[:,4].flatten())
    ax.set_title("E4 Prediction MAE vs STD")
    ax.set_xlabel("MAE [eV]")
    ax.set_ylabel("STD [eV]")
    ax.set_xlim(0, 0.00125)
    ax.set_ylim(0.0, 4.5)
    plt.show()
    plt.close()
    
    # plot mae vs std
    fig, ax = plt.subplots()
    h = ax.scatter(error[:,5].flatten(), std[:,5].flatten())
    ax.set_title("E5 Prediction MAE vs STD")
    ax.set_xlabel("MAE [eV]")
    ax.set_ylabel("STD [eV]")
    ax.set_xlim(0, 0.00125)
    ax.set_ylim(0.0, 4.5)
    plt.show()
    plt.close()
    
    # plot mae vs std
    fig, ax = plt.subplots()
    h = ax.scatter(error[:,6].flatten(), std[:,6].flatten())
    ax.set_title("E6 Prediction MAE vs STD")
    ax.set_xlabel("MAE [eV]")
    ax.set_ylabel("STD [eV]")
    ax.set_xlim(0, 0.00125)
    ax.set_ylim(0.0, 4.5)
    plt.show()
    plt.close()
    """

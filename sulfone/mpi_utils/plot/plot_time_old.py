#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:58:47 2023

@author: chen
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

if __name__ == "__main__":
    res_dir = "../results/al_time/old_write_individual/init_std_05"
    n_model = 4
    
    ############################
    #    Read all time data    #
    ############################
    # MD
    md_to_mg = []
    mg_to_md = []
    prop = []
    # PL
    mg_to_pl = []
    predict = []
    pl_to_mg = []
    update_pl = []
    # ML
    gather_weight = []
    send_weight = []
    # MG
    coord_from_md = []
    coord_to_pl = []
    pred_from_pl = []
    check_qm = []
    coord_to_qm = []
    data_to_ml = []
    pred_to_md = []
    for f in os.listdir(res_dir):
        if "mdtime" in f:
            with open(os.path.join(res_dir, f), 'r') as fh:
                md_time = json.load(fh)
            md_to_mg += md_time['gather']
            mg_to_md += md_time['scatter']
            prop += md_time['prop']
        
        elif "pltime" in f:
            with open(os.path.join(res_dir, f), 'r') as fh:
                pl_time = json.load(fh)
            mg_to_pl += pl_time['bcast']
            predict += pl_time['predict']
            pl_to_mg += pl_time['gather']
            update_pl += pl_time['update']
            
        elif "mltime" in f:
            with open(os.path.join(res_dir, f), 'r') as fh:
                ml_time = json.load(fh)
            gather_weight += ml_time['gather_weight']
            send_weight += ml_time['scatter_weight']
        
        elif "mgtime" in f:
            with open(os.path.join(res_dir, f), 'r') as fh:
                mg_time = json.load(fh)
            coord_from_md += mg_time['coord_from_md']
            coord_to_pl += mg_time['coord_to_pl']
            pred_from_pl += mg_time['pred_from_pl']
            check_qm += mg_time['check_qm']
            coord_to_qm += mg_time['to_qm']
            data_to_ml += mg_time['to_ml']
            pred_to_md += mg_time['pred_to_md']
        
    
    ######################
    #    Plot MD time    #
    ######################
    print("MD Part")
    md_to_mg = np.array(md_to_mg, dtype=float)
    mg_to_md = np.array(mg_to_md, dtype=float)
    mg_to_md_sort = np.sort(mg_to_md)
    prop = np.array(prop, dtype=float)
    print(f"Average time of MD sending message to MG: {np.mean(md_to_mg)}")
    print(f"Average time of MD receiving message from MG: {np.mean(mg_to_md)}")
    print(f"95% cut of time of MD receiving message from MG: {mg_to_md_sort[int(mg_to_md_sort.shape[0] * 0.95)]}")
    print(f"Mode of time of MD receiving message from MG: {st.mode(mg_to_md)[0][0]}")
    print(f"Average trajectory propagation time: {np.mean(prop)}")
    
    # plot histogram of md_to_md distribution
    fig = plt.figure()
    plt.hist(mg_to_md, bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of MD Receiving Message from MG")
    plt.show()
    plt.close()
    
    # (95% cut) plot histogram of md_to_md distribution
    fig = plt.figure()
    plt.hist(mg_to_md_sort[:int(mg_to_md_sort.shape[0] * 0.95)], bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of MD Receiving Message from MG (95% cut)")
    plt.show()
    plt.close()
    
    # plot mg_to_md for first 10000 time steps
    plt.plot(np.arange(0, mg_to_md[:10000].shape[0], dtype=int), mg_to_md[:10000])
    plt.xlabel("Time Step")
    plt.ylabel("Time [s]")
    plt.title("Time of MD Receving Message for 10K Time Steps")
    plt.show()
    plt.close()
    
    # plot mg_to_md for first 100000 time steps
    plt.plot(np.arange(0, mg_to_md[:100000].shape[0], dtype=int), mg_to_md[:100000])
    plt.xlabel("Time Step")
    plt.ylabel("Time [s]")
    plt.title("Time of MD Receving Message for 100K Time Steps")
    plt.show()
    plt.close()
    
    
    ######################
    #    Plot PL time    #
    ######################
    print()
    print("PL Part")
    mg_to_pl = np.array(mg_to_pl, dtype=float)
    mg_to_pl_sort = np.sort(mg_to_pl)
    predict = np.array(predict, dtype=float)
    pl_to_mg = np.array(pl_to_mg, dtype=float)
    update_pl = np.array(update_pl, dtype=float)
    print(f"Average time of PL receving message from MG: {np.mean(mg_to_pl)}")
    print(f"Average time of PL making prediction: {np.mean(predict)}")
    print(f"Average time of PL sending message to MG: {np.mean(pl_to_mg)}")
    print(f"Average time of PL updating models: {np.mean(update_pl)}")
    
    # plot histogram of mg_to_pl distribution
    fig = plt.figure()
    plt.hist(mg_to_pl, bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of PL Receiving Message from MG")
    plt.show()
    plt.close()
    
    # (95% cut) plot histogram of mg_to_pl distribution
    fig = plt.figure()
    plt.hist(mg_to_md_sort[:int(mg_to_pl_sort.shape[0] * 0.95)], bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of PL Receiving Message from MG (95% cut)")
    plt.show()
    plt.close()
    
    # plot mg_to_pl for first 100000 time steps
    plt.plot(np.arange(0, mg_to_pl[:100000].shape[0], dtype=int), mg_to_pl[:100000])
    plt.xlabel("Time Step")
    plt.ylabel("Time [s]")
    plt.title("Time of PL Receving Message for 100K Time Steps")
    plt.show()
    plt.close()
    
    # plot histogram of update_pl distribution
    fig = plt.figure()
    plt.hist(update_pl, bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of PL updating models")
    plt.show()
    plt.close()
    
    ######################
    #    Plot ML time    #
    ######################
    print()
    print("ML Part")
    gather_weight = np.array(gather_weight, dtype=float)
    send_weight = np.array(send_weight, dtype=float)
    print(f"Average time of ML gathering weights: {np.mean(gather_weight)}")
    print(f"Average time of ML sending weights: {np.mean(send_weight)}")
    
    # plot histogram of gather_weight distribution
    fig = plt.figure()
    plt.hist(gather_weight, bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of ML gathering weights")
    plt.show()
    plt.close()
    
    # plot histogram of send_weight distribution
    fig = plt.figure()
    plt.hist(send_weight, bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of ML sending weights")
    plt.show()
    plt.close()
    
    ######################
    #    Plot MG time    #
    ######################
    print()
    print("MG Part")
    coord_from_md = np.array(coord_from_md, dtype=float)
    coord_from_md_sort = np.sort(coord_from_md)
    coord_to_pl = np.array(coord_to_pl, dtype=float)
    pred_from_pl = np.array(pred_from_pl, dtype=float)
    check_qm = np.array(check_qm, dtype=float)
    coord_to_qm = np.array(coord_to_qm, dtype=float)
    data_to_ml = np.array(data_to_ml, dtype=float)
    pred_to_md = np.array(pred_to_md, dtype=float)
    print(f"Average time of MG receiving message from MD: {np.mean(coord_from_md)}")
    print(f"Average time of MG sending message to PL: {np.mean(coord_to_pl)}")
    print(f"Average time of MG receiving message from PL: {np.mean(pred_from_pl)}")
    print(f"Average time of MG checking free QM list: {np.mean(check_qm)}")
    print(f"Average time of MG sending message to QM: {np.mean(coord_to_qm)}")
    print(f"Average time of MG sending message to ML: {np.mean(data_to_ml)}")
    print(f"Average time of MG sending message to MD: {np.mean(pred_to_md)}")
    
    # plot histogram of coord_from_md distribution
    fig = plt.figure()
    plt.hist(coord_from_md, bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of MG receiving data from MD")
    plt.show()
    plt.close()
    
    # (95% cut) plot histogram of coord_from_md_sort distribution
    fig = plt.figure()
    plt.hist(coord_from_md_sort[:int(coord_from_md_sort.shape[0] * 0.95)], bins=100)
    plt.xlabel("Time [s]")
    plt.ylabel("Count")
    plt.title("Time Distribution of MG receiving data from MD (95% cut)")
    plt.show()
    plt.close()
    
    # plot coord_from_md for first 100000 time steps
    plt.plot(np.arange(0, coord_from_md[:100000].shape[0], dtype=int), coord_from_md[:100000])
    plt.xlabel("Time Step")
    plt.ylabel("Time [s]")
    plt.title("Time of MG Receving MD Message for 100K Time Steps")
    plt.show()
    plt.close()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:17:59 2022

@author: Chen Zhou
"""
import numpy as np
import json
import pickle

from pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([0, 1, 2, 3]) #No GPU for prediciton or this main class

from pyNNsMD.nn_pes import NeuralNetPes
import tensorflow as tf

if __name__ == "__main__":
    data_path = "./data/training_70815_random_order_correct.json"

    # read training data: coordinates, energies, gradients
    with open(data_path, 'r') as fh:
        coords, engs, grads = json.load(fh)

    # extract and format list of atoms, np.ndarray of coordinates, energies, gradients
    coords = np.array(coords)
    atoms = coords[:,:,0].tolist()
    coords = np.array(coords[:,:,1:], dtype=float)
    engs=np.array(engs, dtype=float) * 27.21138624598853  # Hatree to eV

    grad_shape = (engs.shape[0], engs.shape[1], coords.shape[1], coords.shape[2])
    grads_all = np.zeros(grad_shape, dtype=float)
    for i in range(0, len(grads)):
        for s in grads[i]:
            # s[0] indicate the state of gradient: 0 -> S0, 1 -> S1, ...
            # s[1] contains the value of gradient
            grads_all[i][int(s[0])] = np.array(s[1], dtype=float) * 27.21138624598853/0.52917721090380  # Hatree to eV
            
    nn = NeuralNetPes("results/eg_model_optimized_extra", mult_nn=2)

    hyper_grads =  {    #Model
                    'general':{
                        'model_type' : 'mlp_eg'
                    },
                    'model':{
                        'use_dropout': False,
                        "dropout": 0.0,
                        'atoms': 38,
                        'states': 7,
                        'depth' : 6,
                        'nn_size' : 10000,   # size of each layer
                        'use_reg_weight' : {'class_name': 'L2', 'config': {'l2': 1e-4}},
                        "activ": {"class_name": "sharp_softplus", "config": {"n": 10.0}},
                        'invd_index' : True,
                        #'activ': 'relu',
                    },
                    'training':{
                        "energy_only": False,
                        "masked_loss": True,
                        "auto_scaling": {"x_mean": True, "x_std": True, "energy_std": True, "energy_mean": True},
                        "loss_weights": {"energy": 1, "force": 1},
                        'learning_rate': 1e-06,
                        "initialize_weights": True,
                        "val_disjoint": True,
                        'normalization_mode' : 1,
                        'epo': 1000,
                        'val_split' : 0.25,
                        'batch_size' : 64,
                        "epostep": 10,
                        "exp_callback": {"use": True, "factor_lr": 0.9, "epomin": 100, "learning_rate_start": 1e-06, "epostep": 20},
                        #'linear_callback' : {'use' : True, 'learning_rate_start' : 1e-6,'learning_rate_stop' : 1e-7, 'epomin' : 100, 'epo': 900},
                        #'log_callback': {'use': True, 'learning_rate_start': 1e-6, 'learning_rate_stop': 1e-7, 'epo': 20, 'epomin': 20, 'epomax': 50},
                        #'step_callback' : {'use': True , 'epoch_step_reduction' : [2000,2000,500,500],'learning_rate_step' :[10 ** -6.0,10 ** -7.0,10 ** -8.0,10 ** -9.0]},
                    }
                    }

    nn.create({'eg': hyper_grads})

    y = {'eg': [engs, grads_all]}

    fitres = nn.fit(coords,
                    y,
                    gpu_dist={'eg': [0, 1, 2, 3]},
                    #gpu_dist= {'e': [0]},
                    proc_async=True,
                    fitmode='training',
                    random_shuffle=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 03:46:57 2023

@author: chen

Function:
    1. Compute and save inversed distance features for analysis.
    2. Random shuffle a feature and evaluate model performance.
"""

import numpy as np
import json, pickle, sys, os, gc

from pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([0, 1, 2, 3]) #No GPU for prediciton or this main class

from pyNNsMD.nn_pes import NeuralNetPes
from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.utils.callbacks import lr_exp_reduction
from pyNNsMD.scaler.energy import MaskedEnergyGradientStandardScaler
from pyNNsMD.utils.loss import get_lr_metric, MaskedScaledMeanAbsoluteError, masked_r2_metric, mask_MeanSquaredError
import tensorflow as tf

if __name__ == "__main__":
    #data_path = "./data/new_training_set0.npy"
    mode = sys.argv[1]
    data_path = sys.argv[2]
    res_path = sys.argv[3]
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    model_hyper = {
                    'use_dropout': False,
                    "dropout": 0.0,
                    'atoms': 38,
                    'states': 7,
                    'depth' : 6,
                    'nn_size' : 10000,   # size of each layer
                    'use_reg_weight' : {'class_name': 'L2', 'config': {'l2': 1e-4}},
                    "activ": {"class_name": "sharp_softplus", "config": {"n": 10.0}},
                    #'activ': {'class_name': "leaky_softplus", "config": {'alpha': 0.03}},
                    'invd_index' : True,
                    #'activ': 'relu',
                    }
    model = EnergyGradientModel(**model_hyper)    # create models
    model.precomputed_features = True
    model.output_as_dict = False
    model.energy_only = True
    
    if mode == "test":
        coords = np.array([[i, i, i] for i in range(1, 11)], dtype=float)
    elif mode == "feature":
        # read training data: coordinates, energies, gradients
        with open(data_path, 'rb') as fh:
            coords = np.load(fh)[:-10000]
    
    if mode != "predict":
        scaler = MaskedEnergyGradientStandardScaler()
        x_scaled, _ = scaler.fit_transform(x=coords, y=None, auto_scale={'x_mean': True, 'x_std': True, 'energy_std': False, 'energy_mean': False})
        feat_x, _ = model.precompute_feature_in_chunks(x_scaled, batch_size=32)
        
        with open(os.path.join(res_path, "feature.npy"), "wb") as fh:
            np.save(fh, feat_x.numpy())
    
    
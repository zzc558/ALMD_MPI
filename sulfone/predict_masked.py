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
    data_path = "./data/testing_23604_random_order_correct.json"

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
            
    nn = NeuralNetPes("results/eg_model_optimized_for_test", mult_nn=4)
    nn.load()

    res = nn.predict(coords)

    with open("./data/eg_model_optimized_for_test_prediction_results", 'wb') as fh:
        pickle.dump(res, fh)

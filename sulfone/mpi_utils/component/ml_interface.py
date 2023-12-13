#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:19:21 2023

@author: chen
"""
#from tensorflow.keras import backend as K
import numpy as np
import tensorflow.keras as ks
from mpi4py import MPI

#def masked_MAE(y_true, y_pred):
#    mask = K.cast(K.not_equal(y_true, 0.0), dtype=float)
#    return K.sum(K.abs(y_true*mask - y_pred*mask))/(K.sum(mask))

def masked_MAE(y_true, y_pred):
    mask = np.not_equal(y_true, 0.0).astype(dtype=float)
    return np.sum(np.abs(y_true*mask - y_pred*mask))/(np.sum(mask))

def masked_r2(y_true, y_pred):
    mask = np.not_equal(y_true, 0.0).astype(dtype=float)
    ss_res = np.sum(np.square(y_true*mask - y_pred*mask))
    ss_tot = np.sum(np.square((y_true * mask - np.sum(y_true * mask)/np.sum(mask)) * mask))
    return 1 - ss_res / (ss_tot + np.finfo(float).eps)

class MPICallback(ks.callbacks.Callback):
    def __init__(self, req):
        super(MPICallback, self).__init__()
        self.req = req
        self.status = MPI.Status()
        
    def on_epoch_end(self, epoch, logs=None):
        if self.req.Test(self.status):
            self.model.stop_training = True
            print()
            print('New data arrived. Retraining restarts...')
            print()

class MLForAl(object):
    def __init__(self, method, kwargs):
        if method == 'nn':
            from component.nn_mpi import NNforMPI
            self.method = NNforMPI(**kwargs)
        
    def retrain(self, request):
        self.method.retrain(request)
    
    def add_trainingset(self, coords, energies, gradients):
        self.method.add_trainingset(coords, energies, gradients)
    
    def update(self, weight_array):
        self.method.update(weight_array)
    
    def predict(self, coords):
        results = self.method.predict(coords)
        return results
    
    def get_weight(self):
        return self.method.get_weight()
    
    def get_num_weight(self):
        return self.method.get_num_weight()
    
    def save_progress(self):
        """save retrain history"""
        self.method.save_progress()

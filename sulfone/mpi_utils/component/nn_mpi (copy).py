#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:18:50 2023

@author: chen
"""
import numpy as np
import json
import pickle
import os, gc
import shutil
from mpi4py import MPI

from pyNNsMD.nn_pes_src.device import set_gpu
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler, MaskedEnergyGradientStandardScaler
from pyNNsMD.utils.loss import get_lr_metric, ScaledMeanAbsoluteError, r2_metric, ZeroEmptyLoss, MaskedScaledMeanAbsoluteError, masked_r2_metric, mask_MeanSquaredError

HtoEv = 27.21138624598853
AToBohr = 1.889725989

class MPICallback(ks.callbacks.Callback):
    def __init__(self, req):
        super(MPICallback, self).__init__()
        self.req = req
        self.status = MPI.Status()
        self.patience = 2
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.Inf
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss", None)
        if self.req.Test(self.status):
            self.model.stop_training = True
            print()
            print('New data arrived. Retraining restarts...')
            print()
        elif val_loss != None:
            if np.less(val_loss, self.best):
                self.wait = 0
                self.best = val_loss
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
                    print("Early stopping: validation loss is getting higher!")

class NNforMPI(object):
    def __init__(self, gpu_index, model_index, model_dir, model_name, hyper, source, mode='prediction'):
        # TODO: skip NeuralNetPes and work with training_mlp_eg directly. Switch to new version.

        set_gpu([gpu_index,])
        # initialize NeuralNetPes and load hyperparameters
        self.model_dir = model_dir
        self.model_name = model_name
        self.hyper = hyper
        self.model_index = model_index
        self.val_split = hyper['retraining']['val_split']
        self._model = EnergyGradientModel(**self.hyper['model'])    # create models
        self._scaler = MaskedEnergyGradientStandardScaler()    # create scalar
        self._model_setup()    # load weights for model and scaler

        if mode == 'retrain':
            # user input to initialize 
            pass
        elif mode = 'prediction':
            # user input to build PL models
            pass

            

    def add_trainingset(self, new_coord, new_energy, new_force):
        # do somthing
        save_progress(model_path)
        
        pass

    def _add_hist(self, hist_new):
        self.hist['energy_mean_absolute_error'] += hist_new['energy_mean_absolute_error']
        self.hist['energy_r2'] += hist_new['energy_masked_r2_metric']
        self.hist['val_energy_mean_absolute_error'] += hist_new['val_energy_mean_absolute_error']
        self.hist['val_energy_r2'] += hist_new['val_energy_masked_r2_metric']
        self.hist['force_mean_absolute_error'] += hist_new['force_mean_absolute_error']
        self.hist['force_r2'] += hist_new['force_masked_r2_metric']
        self.hist['val_force_mean_absolute_error'] += hist_new['val_force_mean_absolute_error']
        self.hist['val_force_r2'] += hist_new['val_force_masked_r2_metric']
        self.hist['num_epoch'].append(len(hist_new['energy_mean_absolute_error']))

        #with open(self.hist_path, 'w') as fh:
        #    json.dump(self.hist, fh)
            
    def _fit_model(self, req):
        fit_hyper = self.hyper['retraining']
        batch_size = fit_hyper['batch_size']
        learning_rate = fit_hyper['learning_rate']
        energy_only = fit_hyper['energy_only']
        loss_weights = fit_hyper['loss_weights']
        epo = fit_hyper['epo']
        epostep = fit_hyper['epostep']
        #set_gpu([self.gpu_list[i],])
        self._model.precomputed_features = True
        self._model.output_as_dict = True
        cbks = [MPICallback(req)]

        # scale x, y
        self._scaler.fit(self.coord, [self.energy, self.force])
        x_rescale, y_rescale = self._scaler.transform(self.coord, [self.energy, self.force])
        y1, y2 = y_rescale

        # Model + Model precompute layer +feat
        feat_x, feat_grad = self._model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)
        # Finding Normalization
        feat_x_mean, feat_x_std = self._model.set_const_normalization_from_features(feat_x,normalization_mode=1)

        # Train Test split
        #i_val = []
        #for j in range(0, self.coords.shape[0]):
        #    if j not in self.index[i]:
        #        i_val.append(j)
        #i_val = np.array(i_val, dtype=int)
        xtrain = [feat_x[self.i_train], feat_grad[self.i_train]]
        ytrain = [y1[self.i_train], y2[self.i_train]]
        xval = [feat_x[self.i_val], feat_grad[self.i_val]]
        yval = [y1[self.i_val], y2[self.i_val]]

        # Setting constant feature normalization
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        lr_metric = get_lr_metric(optimizer)

        mae_energy = MaskedScaledMeanAbsoluteError(scaling_shape=self._scaler.energy_std.shape)
        mae_force = MaskedScaledMeanAbsoluteError(scaling_shape=self._scaler.gradient_std.shape)
        mae_energy.set_scale(self._scaler.energy_std)
        mae_force.set_scale(self._scaler.gradient_std)
        train_metrics = {'energy': [mae_energy, lr_metric, masked_r2_metric],
                        'force': [mae_force, lr_metric, masked_r2_metric]}
        if energy_only:
            train_loss = {'energy': mask_MeanSquaredError, 'force' : ZeroEmptyLoss()}
        else:
            train_loss = {'energy': mask_MeanSquaredError, 'force': mask_MeanSquaredError}

        self._model.compile(optimizer=optimizer,
                      loss=train_loss, loss_weights=loss_weights,
                      metrics=train_metrics)
        self._scaler.print_params_info()
        print("Info: Using feature-scale", feat_x_std.shape, ":", feat_x_std)
        print("Info: Using feature-offset", feat_x_mean.shape, ":", feat_x_mean)
        print("Info: Feature data type: ",  feat_x.dtype, feat_grad.dtype)
        print("Info: Target data type: ", ytrain[0].dtype, ytrain[1].dtype)

        print("")
        print("Start fit.")
        self._model.summary()
        hist = self._model.fit(x=xtrain, y={'energy': ytrain[0], 'force': ytrain[1]}, epochs=epo, batch_size=batch_size,
                         callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, {'energy': yval[0], 'force': yval[1]}), verbose=2)
        print("End fit.")
        print("")
        
        # remove tensors to free gpu memory
        del xtrain, ytrain, xval, yval, feat_x, feat_grad, y1, y2, y_rescale, x_rescale
        gc.collect()
        
        self._model.precomputed_features = False
        self._model.output_as_dict = False
        outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
        self._add_hist(outhist)
        weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}.h5')
        os.remove(weight_path)
        self._model.save_weights(weight_path)
        self._scaler.save(os.path.join(self.model_dir, self.model_name, f'scaler_v{self.model_index}.json'))
        
    def retrain(self, req):
        
        # add functions to retrain zour model
        # please use the following call backs cbks = [MPICallback(req), zour own]  for tf models
        #self._add_trainingset(coords, energies, gradients)
        self._fit_model(req)
        #for i in range(0, self.n_models):
        #    self._fit_models(i)
        #idx = [(i,) for i in range(0, self.n_models)]
        #with Pool(self.n_models) as pool:
        #    pool.starmap_async(self._fit_models, idx)
        
    def save_hist(self):
        # save model and scalar weights
        #weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}.h5')
        #os.remove(weight_path)
        #self._model.save_weights(weight_path)
        #self._scaler.save(os.path.join(self.model_dir, self.model_name, f'scaler_v{self.model_index}.json'))
        
        # save retrain history
        with open(self.hist_path, 'w') as fh:
            json.dump(self.hist, fh)

    def predict(self, x):
        #set_gpu([self.gpu_list[i],])
        batch_size = self.hyper['general']['batch_size_predict']
        x_scaled = self._scaler.transform(x=x)[0]
        res = self._model.predict(x_scaled, batch_size=batch_size)
        return self._scaler.inverse_transform(y=res)[1]
    
    #def predict(self, coords):
    #    eng_pred = []
    #    force_pred = []
    #    batch_size = self.hyper['general']['batch_size_predict']
    #    
    #    for i in range(0, self.n_models):
    #        x_scaled = self._scaler[i].transform(x=coords)[0]
    #        res = self._models[i].predict(x_scaled, batch_size=batch_size)
    #        res = self._scaler[i].inverse_transform(y=res)[1]
    #        eng_pred.append(res[0])
    #        force_pred.append(res[1])
    #    print("Prediction finished.")
    #    return np.array(eng_pred,dtype=float), np.array(force_pred, dtype=float)


    
    def update(self):
        #rm_cmd = 'rm '
        #cp_cmd = 'cp '
        #for i in range(0, self.n_models):
        #    rm_cmd += os.path.join(self.model_dir, self.model_name, f'weights_v{i}.h5') + ' '\
        #        + os.path.join(self.model_dir, self.model_name, f'scaler_v{i}.json') + ' '
        #    cp_cmd += os.path.join(self.source, f'weights_v{i}.h5') + ' '\
        #        + os.path.join(self.source, f'scaler_v{i}.json') + ' '

        #cp_cmd += os.path.join(self.model_dir, self.model_name)
        #os.system(rm_cmd)
        #os.system(cp_cmd)
        scalar_path = os.path.join(self.model_dir, self.model_name, f'scaler_v{self.model_index}.json')
        scalar_source = os.path.join(self.source, self.model_name, f'scaler_v{self.model_index}.json')
        weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}.h5')
        weight_source = os.path.join(self.source, self.model_name, f'weights_v{self.model_index}.h5')
        
        shutil.copyfile(src=scalar_source, dst=scalar_path)
        shutil.copyfile(src=weight_source, dst=weight_path)

        self._model_setup()
        #eng, force = self.predict(self.test_set[0])
        #return eng, force
        #eng_mae, eng_r2, std, force_mae, force_r2 = self._evaluate_predictions(eng, force, self.test_set[1], self.test_set[2])

        #test_res_path = os.path.join(self.model_dir, self.model_name, 'test_results.json')
        #if os.path.exists(test_res_path):
        #    with open(test_res_path, 'r') as fh:
        #        test_res = json.load(fh)
        #else:
        #    test_res = {
        #        'eng_mae': [],
        #        'eng_r2': [],
        #        'force_mae': [],
        #        'force_r2': [],
        #        'std': []
        #        }
        #test_res['eng_mae'] += [float(eng_mae.numpy()),]
        #test_res['eng_r2'] += [float(eng_r2.numpy()),]
        #test_res['force_mae'] += [float(force_mae.numpy()),]
        #test_res['force_r2'] += [float(force_r2.numpy()),]
        #test_res['std'] += std.tolist()
        #with open(test_res_path, 'w') as fh:
        #    json.dump(test_res, fh)

def masked_MAE(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), dtype=float)
    return K.sum(K.abs(y_true*mask - y_pred*mask))/(K.sum(mask))

def evaluate_predictions(self, eng_pred, force_pred, eng_true, force_true):
    std = np.std(eng_pred, axis=0, ddof=1)
    eng_mean = np.mean(eng_pred, axis=0)
    force_mean = np.mean(force_pred, axis=0)
    eng_mae = masked_MAE(eng_true, eng_mean)
    force_mae = masked_MAE(force_true, force_mean)
    eng_r2 = masked_r2_metric(eng_true, eng_mean)
    force_r2 = masked_r2_metric(force_true, force_mean)
    return eng_mae, eng_r2, std, force_mae, force_r2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:30:23 2023

@author: chen
"""

import numpy as np
import json
import pickle
import os
import shutil
from multiprocessing.pool import Pool

from pyNNsMD.nn_pes_src.device import set_gpu
from pyNNsMD.nn_pes import NeuralNetPes
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler, MaskedEnergyGradientStandardScaler
from pyNNsMD.utils.loss import get_lr_metric, ScaledMeanAbsoluteError, r2_metric, ZeroEmptyLoss, MaskedScaledMeanAbsoluteError, masked_r2_metric, mask_MeanSquaredError

HtoEv = 27.21138624598853
AToBohr = 1.889725989

class NNforMPI(object):
    def __init__(self, gpu_list, model_dir, n_models, model_name, hyper, source, test_path, mode='prediction'):
        # TODO: skip NeuralNetPes and work with training_mlp_eg directly. Switch to new version.
        
        # initialize NeuralNetPes and load hyperparameters
        self.model_dir = model_dir
        self.model_name = model_name
        self.n_models = n_models
        self.source = source    # for passive learner only. path to the model of ML processes
        #self.nn = NeuralNetPes(model_dir, n_models)
        #self.nn.load(model_name)
        #self.nn.update(hyper)
        self.hyper = hyper
        self.gpu_list = gpu_list
        self.val_split = hyper['retraining']['val_split']
        self._models = [EnergyGradientModel(**self.hyper['model']) for i in range(0, self.n_models)]    # create models
        self._scaler = [MaskedEnergyGradientStandardScaler() for i in range(0, self.n_models)]    # create scalar
        self._model_setup()    # load weights for model and scaler

        set_gpu([self.gpu_list[0],])
        
        if mode == 'retrain':
            # keep records of training set for each model
            self.training_set_path = [os.path.join(self.model_dir, self.model_name, 'data_x'), os.path.join(self.model_dir, self.model_name, 'data_y')]
            with open(self.training_set_path[0], 'rb') as fh:
                self.coords = pickle.load(fh)
            with open(self.training_set_path[1], 'rb') as fh:
                self.energy, self.force = pickle.load(fh)
            # x and y data for all models are stored together to save space
            # need to load index and rebuild data for each model
            self.index = []
            for i in range(0, self.n_models):
                self.index.append(np.load(os.path.join(self.model_dir, self.model_name, 'index', f'train_val_idx_v{i}.npy')))

            self.hist = {
                'energy_mean_absolute_error': [[],]*self.n_models,
                'energy_r2': [[],]*self.n_models,
                'val_energy_mean_absolute_error': [[],]*self.n_models,
                'val_energy_r2': [[],]*self.n_models,
                'force_mean_absolute_error': [[],]*self.n_models,
                'force_r2': [[],]*self.n_models,
                'val_force_mean_absolute_error': [[],]*self.n_models,
                'val_force_r2': [[],]*self.n_models,
                }
        else:
            with open(test_path, 'r') as fh:
                c, e, g = json.load(fh)
            c = np.array(c)
            c = np.array(c[:,:,1:], dtype=float)
            e = np.array(e, dtype=float) * HtoEv
            g_shape = (e.shape[0], e.shape[1], c.shape[1], c.shape[2])
            g_all = np.zeros(g_shape, dtype=float)
            for i in range(0, len(g)):
                for s in g[i]:
                    # s[0] indicate the state of gradient: 0 -> S0, 1 -> S1, ...
                    # s[1] contains the value of gradient
                    g_all[i][int(s[0])] = np.array(s[1], dtype=float) * HtoEv * AToBohr
            self.test_set = [c, e, g_all]
                
    def _model_setup(self):
        energy_only = self.hyper['retraining']['energy_only']
        for i in range(0, self.n_models):
            # load pretrained model weights
            self._models[i].load_weights(os.path.join(self.model_dir, self.model_name, f'weights_v{i}.h5'))
            self._models[i].precomputed_features = False
            self._models[i].output_as_dict = False
            self._models[i].energy_only = energy_only
            # load scaler weights
            self._scaler[i].load(os.path.join(self.model_dir, self.model_name, f"scaler_v{i}.json"))
        
    def _add_trainingset(self, new_coord, new_energy, new_force):
        idx = np.array(range(self.coords.shape[0], self.coords.shape[0]+new_coord.shape[0]), dtype=int)
        self.coords = np.concatenate((self.coords, new_coord), axis=0)
        self.energy = np.concatenate((self.energy, new_energy), axis=0)
        self.force = np.concatenate((self.force, new_force), axis=0)
        
        np.random.shuffle(idx)
        new_size = int((1-self.val_split)*new_coord.shape[0])
        for i in range(0, self.n_models):
            self.index[i] = np.concatenate((self.index[i], np.random.choice(idx, size=new_size, replace=False)), axis=0)
            
        with open(os.path.join(self.model_dir, self.model_name, 'new_training_set.npy'), 'wb') as fh:
            np.save(fh, self.coords)
            np.save(fh, self.energy)
            np.save(fh, self.force)
            np.save(fh, np.array(self.index, dtype=int))

    def _add_hist(self, hist_new, i):
        hist_path = os.path.join(self.model_dir, self.model_name, 'retrain_history.json')
        
        self.hist['energy_mean_absolute_error'][i] += hist_new['energy_mean_absolute_error']
        self.hist['energy_r2'][i] += hist_new['energy_masked_r2_metric']
        self.hist['val_energy_mean_absolute_error'][i] += hist_new['val_energy_mean_absolute_error']
        self.hist['val_energy_r2'][i] += hist_new['val_energy_masked_r2_metric']
        self.hist['force_mean_absolute_error'][i] += hist_new['force_mean_absolute_error']
        self.hist['force_r2'][i] += hist_new['force_masked_r2_metric']
        self.hist['val_force_mean_absolute_error'][i] += hist_new['val_force_mean_absolute_error']
        self.hist['val_force_r2'][i] += hist_new['val_force_masked_r2_metric']
        
        with open(hist_path, 'w') as fh:
            json.dump(self.hist, fh)
    
    def _fit_models(self, i):
        fit_hyper = self.hyper['retraining']
        batch_size = fit_hyper['batch_size']
        learning_rate = fit_hyper['learning_rate']
        energy_only = fit_hyper['energy_only']
        loss_weights = fit_hyper['loss_weights']
        epo = fit_hyper['epo']
        epostep = fit_hyper['epostep']
        set_gpu([self.gpu_list[i],])
        self._models[i].precomputed_features = True
        self._models[i].output_as_dict = True
        cbks = []
        
        # scale x, y
        self._scaler[i].fit(self.coords, [self.energy, self.force])
        x_rescale, y_rescale = self._scaler[i].transform(self.coords, [self.energy, self.force])
        y1, y2 = y_rescale
        
        # Model + Model precompute layer +feat
        feat_x, feat_grad = self._models[i].precompute_feature_in_chunks(x_rescale, batch_size=batch_size)
        # Finding Normalization
        feat_x_mean, feat_x_std = self._models[i].set_const_normalization_from_features(feat_x,normalization_mode=1)
        
        # Train Test split
        i_val = []
        for j in range(0, self.coords.shape[0]):
            if j not in self.index[i]:
                i_val.append(j)
        i_val = np.array(i_val, dtype=int)
        xtrain = [feat_x[self.index[i]], feat_grad[self.index[i]]]
        ytrain = [y1[self.index[i]], y2[self.index[i]]]
        xval = [feat_x[i_val], feat_grad[i_val]]
        yval = [y1[i_val], y2[i_val]]
        
        # Setting constant feature normalization
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        lr_metric = get_lr_metric(optimizer)
        
        mae_energy = MaskedScaledMeanAbsoluteError(scaling_shape=self._scaler[i].energy_std.shape)
        mae_force = MaskedScaledMeanAbsoluteError(scaling_shape=self._scaler[i].gradient_std.shape)
        mae_energy.set_scale(self._scaler[i].energy_std)
        mae_force.set_scale(self._scaler[i].gradient_std)
        train_metrics = {'energy': [mae_energy, lr_metric, masked_r2_metric],
                        'force': [mae_force, lr_metric, masked_r2_metric]}
        if energy_only:
            train_loss = {'energy': mask_MeanSquaredError, 'force' : ZeroEmptyLoss()}
        else:
            train_loss = {'energy': mask_MeanSquaredError, 'force': mask_MeanSquaredError}
            
        self._models[i].compile(optimizer=optimizer,
                      loss=train_loss, loss_weights=loss_weights,
                      metrics=train_metrics)
        self._scaler[i].print_params_info()
        print("Info: Using feature-scale", feat_x_std.shape, ":", feat_x_std)
        print("Info: Using feature-offset", feat_x_mean.shape, ":", feat_x_mean)
        print("Info: Feature data type: ",  feat_x.dtype, feat_grad.dtype)
        print("Info: Target data type: ", ytrain[0].dtype, ytrain[1].dtype)
        
        print("")
        print("Start fit.")
        self._models[i].summary()
        hist = self._models[i].fit(x=xtrain, y={'energy': ytrain[0], 'force': ytrain[1]}, epochs=epo, batch_size=batch_size,
                         callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, {'energy': yval[0], 'force': yval[1]}), verbose=2)
        print("End fit.")
        print("")
        self._models[i].precomputed_features = False
        self._models[i].output_as_dict = False
        outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
        self._add_hist(outhist, i)
        weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{i}.h5')
        os.system(f'rm {weight_path}')
        self._models[i].save_weights(weight_path)
        self._scaler[i].save(os.path.join(self.model_dir, self.model_name, f'scaler_v{i}.json'))
        
    def retrain(self, coords, energies, gradients):
        self._add_trainingset(coords, energies, gradients)
        for i in range(0, self.n_models):
            self._fit_models(i)
        #idx = [(i,) for i in range(0, self.n_models)]
        #with Pool(self.n_models) as pool:
        #    pool.starmap_async(self._fit_models, idx)
        
    def _predict_model(self, x, i):
        set_gpu([self.gpu_list[i],])
        batch_size = self.hyper['general']['batch_size_predict']
        x_scaled = self._scaler[i].transform(x=x)[0]
        res = self._models[i].predict(x_scaled, batch_size=batch_size)
        return self._scaler[i].inverse_transform(y=res)
    
    #def predict(self, coords):
    #    items = [(coords, i) for i in range(0, self.n_models)]
    #    eng_pred = []
    #    force_pred = []
    #    with Pool(self.n_models) as pool:
    #        res = pool.starmap_async(self._predict_model, items)
    #        for r in res.get():
    #            eng_pred.append(r[0])
    #            force_pred.append(r[1])
    #    print("Prediction done.")
    #    return np.array(eng_pred,dtype=float), np.array(force_pred, dtype=float)

    def predict(self, coords):
        eng_pred = []
        force_pred = []
        batch_size = self.hyper['general']['batch_size_predict']
        for i in range(0, self.n_models):
            x_scaled = self._scaler[i].transform(x=coords)[0]
            res = self._models[i].predict(x_scaled, batch_size=batch_size)
            res = self._scaler[i].inverse_transform(y=res)[1]
            eng_pred.append(res[0])
            force_pred.append(res[1])
        print("Prediction finished.")
        return np.array(eng_pred,dtype=float), np.array(force_pred, dtype=float)

    def _evaluate_predictions(self, eng_pred, force_pred, eng_true, force_true):
        std = np.std(eng_pred, axis=0, ddof=1)
        eng_mean = np.mean(eng_pred, axis=0)
        force_mean = np.mean(force_pred, axis=0)
        eng_mae = masked_MAE(eng_true, eng_mean)
        force_mae = masked_MAE(force_true, force_mean)
        eng_r2 = masked_r2_metric(eng_true, eng_mean)
        force_r2 = masked_r2_metric(force_true, force_mean)
        return eng_mae, eng_r2, std, force_mae, force_r2
    
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

        for i in range(0, self.n_models):
            scalar_path = os.path.join(self.model_dir, self.model_name, f'scaler_v{i}.json')
            scalar_source = os.path.join(self.source, self.model_name, f'scaler_v{i}.json')
            weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{i}.h5')
            weight_source = os.path.join(self.source, self.model_name, f'weights_v{i}.h5')
            
            shutil.copyfile(src=scalar_source, dst=scalar_path)
            shutil.copyfile(src=weight_source, dst=weight_path)

        self._model_setup()
        eng, force = self.predict(self.test_set[0])
        eng_mae, eng_r2, std, force_mae, force_r2 = self._evaluate_predictions(eng, force, self.test_set[1], self.test_set[2])
        
        test_res_path = os.path.join(self.model_dir, self.model_name, 'test_results.json')
        if os.path.exists(test_res_path):
            with open(test_res_path, 'r') as fh:
                test_res = json.load(fh)
        else:
            test_res = {
                'eng_mae': [],
                'eng_r2': [],
                'force_mae': [],
                'force_r2': [],
                'std': []
                }
        test_res['eng_mae'] += [float(eng_mae.numpy()),]
        test_res['eng_r2'] += [float(eng_r2.numpy()),]
        test_res['force_mae'] += [float(force_mae.numpy()),]
        test_res['force_r2'] += [float(force_r2.numpy()),]
        test_res['std'] += std.tolist()
        with open(test_res_path, 'w') as fh:
            json.dump(test_res, fh)       
    
def masked_MAE(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), dtype=float)
    return K.sum(K.abs(y_true*mask - y_pred*mask))/(K.sum(mask))

#def eval_std(data_array, threshold):
#    std = np.std(data_array, axis=0, ddof=1)
#    return np.where((std > threshold).any(axis=1))[0]
    #not_pass = []
    #for i in range(0, std.shape):
    #    if (std > threshold).any():
    #        not_pass.append(i)
    #return not_pass
    
#def retrain(self, coords, energies, gradients):
#    with open(self.training_set_path[0], 'rb') as fh:
#        x = pickle.load(fh)
#    with open(self.training_set_path[1], 'rb') as fh:
#        y = pickle.load(fh)
#    
#    x = np.concatenate((x, coords), axis=0)
#    energies = np.concatenate((y[0], energies), axis=0)
#    gradients = np.concatenate((y[1], gradients), axis=0)
#    y = {'eg': [energies, gradients]}
#    
#    fitres = self.nn.fit(x,
#                    y,
#                    gpu_dist={'eg': self.gpu_list},
#                    proc_async=True,
#                    fitmode='retraining',
#                    random_shuffle=True)
#    self._read_hist()

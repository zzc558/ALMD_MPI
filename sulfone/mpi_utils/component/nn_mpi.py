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
from multiprocessing import Process

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
        self.mode = mode

        if self.mode == 'retrain':
            #self.gpu_index = gpu_index
            weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}')
            if os.path.exists(weight_path):
                with open(weight_path, 'rb') as fh:
                    self.model_weight = pickle.load(fh)
            else:
                with tf.device('/CPU:0'):
                    model = EnergyGradientModel(**self.hyper['model'])    # create models
                    model.load_weights(os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}.h5'))
                    self.model_weight = model.get_weights()
                    del model
                    gc.collect()
            self._scaler = MaskedEnergyGradientStandardScaler()    # create scalar
            self._scaler.load(os.path.join(self.model_dir, self.model_name, f"scaler_v{self.model_index}.json"))
            try:
                # load saved training set
                with open(os.path.join(self.model_dir, self.model_name, f'new_training_set{self.model_index}.npy'), 'rb') as fh:
                    self.coord = np.load(fh)
                    self.energy = np.load(fh)
                    self.force = np.load(fh)
                    self.i_train = np.load(fh)
                    self.i_val = np.load(fh)
            except:
                # load initial training set for each model
                self.training_set_path = [os.path.join(self.model_dir, self.model_name, 'data_x'), os.path.join(self.model_dir, self.model_name, 'data_y')]
                with open(self.training_set_path[0], 'rb') as fh:
                    self.coord = pickle.load(fh)
                with open(self.training_set_path[1], 'rb') as fh:
                    self.energy, self.force = pickle.load(fh)
                # x and y data for all models are stored together to save space
                # need to load index and rebuild data for each model
                self.i_train = np.load(os.path.join(self.model_dir, self.model_name, 'index', f'train_val_idx_v{self.model_index}.npy'))
                self.i_val = np.array([i for i in range(0, self.coord.shape[0]) if i not in self.i_train], dtype=int)

            self.hist_path = os.path.join(self.model_dir, self.model_name, f'retrain_history{self.model_index}.json')
            try:
                with open(self.hist_path, 'r') as fh:
                    self.hist = json.load(fh)
            except:
                self.hist = {
                    'energy_mean_absolute_error': [],
                    'energy_r2': [],
                    'val_energy_mean_absolute_error': [],
                    'val_energy_r2': [],
                    'force_mean_absolute_error': [],
                    'force_r2': [],
                    'val_force_mean_absolute_error': [],
                    'val_force_r2': [],
                    'num_epoch': []
                    }
        else:
            #set_gpu([gpu_index,])
            self._model = EnergyGradientModel(**self.hyper['model'])    # create models
            self._scaler = MaskedEnergyGradientStandardScaler()    # create scalar
            self._model_setup()    # load weights for model and scaler
            
    def _model_setup(self):
        weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}')
        if os.path.exists(weight_path):
            with open(weight_path, 'rb') as fh:
                self._model.set_weights(pickle.load(fh))
        else:
            self._model.load_weights(os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}.h5'))
        self._model.precomputed_features = False
        self._model.output_as_dict = False
        self._model.energy_only = self.hyper['retraining']['energy_only']
        self._scaler.load(os.path.join(self.model_dir, self.model_name, f"scaler_v{self.model_index}.json"))

    def add_trainingset(self, new_coord, new_energy, new_force):
        idx = np.array(range(self.coord.shape[0], self.coord.shape[0]+new_coord.shape[0]), dtype=int)
        np.random.shuffle(idx)
        new_size = int((1-self.val_split)*new_coord.shape[0])
        new_train = np.random.choice(idx, size=new_size, replace=False)
        new_val = np.array([i for i in idx if i not in new_train], dtype=int)
        self.i_train = np.concatenate((self.i_train, new_train), axis=0)
        self.i_val = np.concatenate((self.i_val, new_val), axis=0)
        self.coord = np.concatenate((self.coord, new_coord), axis=0)
        self.energy = np.concatenate((self.energy, new_energy), axis=0)
        self.force = np.concatenate((self.force, new_force), axis=0)
        assert self.coord.shape[0] == self.energy.shape[0] and self.coord.shape[0] == self.force.shape[0], "Check training increment at _add_trainingset"
        
        
        #self.coord_val = np.concatenate((self.coord_val, new_coord[i_val]), axis=0)
        #self.energy_val = np.concatenate((self.energy_val, new_energy[i_val]), axis=0)
        #self.force_val = np.concatenate((self.force_val, new_force[i_val]), axis=0)
        #assert self.coord_val.shape[0] == self.energy_val.shape[0] and self.coord_val.shape[0] == self.force_val.shape[0], "Check validation increment at _add_trainingset"

        #with open(os.path.join(self.model_dir, self.model_name, f'new_training_set{self.model_index}.npy'), 'wb') as fh:
        #    np.save(fh, self.coord)
        #    np.save(fh, self.energy)
        #    np.save(fh, self.force)
        #    np.save(fh, self.i_train)
        #    np.save(fh, self.i_val)

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
        tf.keras.backend.clear_session()
        #set_gpu([self.gpu_index,])
        fit_hyper = self.hyper['retraining']
        batch_size = fit_hyper['batch_size']
        learning_rate = fit_hyper['learning_rate']
        energy_only = fit_hyper['energy_only']
        loss_weights = fit_hyper['loss_weights']
        epo = fit_hyper['epo']
        epostep = fit_hyper['epostep']
        #set_gpu([self.gpu_list[i],])
        model = EnergyGradientModel(**self.hyper['model'])    # create models
        model.precomputed_features = True
        model.output_as_dict = True
        model.energy_only = energy_only
        model.set_weights(self.model_weight)
        
        cbks = [MPICallback(req),]

        # scale x, y
        self._scaler.fit(self.coord, [self.energy, self.force])
        x_rescale, y_rescale = self._scaler.transform(self.coord, [self.energy, self.force])
        y1, y2 = y_rescale

        # Model + Model precompute layer +feat
        feat_x, feat_grad = model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)
        # Finding Normalization
        feat_x_mean, feat_x_std = model.set_const_normalization_from_features(feat_x,normalization_mode=1)

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
        
        print(f"Info: shape of training input feature is {feat_x[self.i_train].shape}")
        
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

        model.compile(optimizer=optimizer,
                      loss=train_loss, loss_weights=loss_weights,
                      metrics=train_metrics)
        self._scaler.print_params_info()
        print("Info: Using feature-scale", feat_x_std.shape, ":", feat_x_std)
        print("Info: Using feature-offset", feat_x_mean.shape, ":", feat_x_mean)
        print("Info: Feature data type: ",  feat_x.dtype, feat_grad.dtype)
        print("Info: Target data type: ", ytrain[0].dtype, ytrain[1].dtype)

        print("")
        print("Start fit.")
        model.summary()
        hist = model.fit(x=xtrain, y={'energy': ytrain[0], 'force': ytrain[1]}, epochs=epo, batch_size=batch_size,
                         callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, {'energy': yval[0], 'force': yval[1]}), verbose=2)
        print("End fit.")
        print("")
        
        # remove tensors to free gpu memory
        del xtrain, ytrain, xval, yval, feat_x, feat_grad, y1, y2, y_rescale, x_rescale, self.model_weight
        gc.collect()
        
        # update model weights
        with tf.device('/CPU:0'):
            self.model_weight = model.get_weights()
        del model
        gc.collect()
        
        #self._model.precomputed_features = False
        #self._model.output_as_dict = False
        outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
        self._add_hist(outhist)
        #weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}.h5')
        #os.remove(weight_path)
        #self._model.save_weights(weight_path)
        #self._scaler.save(os.path.join(self.model_dir, self.model_name, f'scaler_v{self.model_index}.json'))
        
    def retrain(self, req):
        #self._add_trainingset(coords, energies, gradients)
        self._fit_model(req)
        
        #p = Process(target=self._fit_model, args=(req,))
        #p.start()
        #p.join()
        #for i in range(0, self.n_models):
        #    self._fit_models(i)
        #idx = [(i,) for i in range(0, self.n_models)]
        #with Pool(self.n_models) as pool:
        #    pool.starmap_async(self._fit_models, idx)
        
    def get_num_weight(self):
        model_weight = self._model.get_weights()
        # count the total number of weights of model
        num_weight = 0
        for w in model_weight:
            s = w.shape
            num_weight += s[0] if len(s)==1 else s[0]*s[1]
        # add the number of weight of scalar
        num_weight += 2    # energy mean and energy std
        num_weight += self._scaler.energy_mean.flatten().shape[0]
        num_weight += self._scaler.energy_std.flatten().shape[0]
        num_weight += self._scaler.gradient_std.flatten().shape[0]
        del model_weight
        gc.collect()
        return num_weight
        
    def get_weight(self):
        # format of weight array that is sent to PL process: [number of layers, number of weights of layer 0, number of weights of layer 1, ..., weights of layer 0, weights of layer 1, ..., scalar weights...]
        #weight_list = self._model.get_weights()
        #weight_array = np.empty((len(weight_list)+1,), dtype=float)
        #weight_array[0] = float(len(weight_list))
        weight_array = np.array([], dtype=float)
        for i in range(0, len(self.model_weight)):
            #weight_array[i+1] = float(weight_list[i].shape[0])
            weight_array = np.concatenate((weight_array, self.model_weight[i].flatten()), axis=0)
        # add scalar parameters to the weight array
        weight_array = np.concatenate((weight_array, self._scaler.x_mean.flatten(), self._scaler.x_std.flatten(), \
                                       self._scaler.energy_mean.flatten(), self._scaler.energy_std.flatten(), self._scaler.gradient_std.flatten()), axis=0)
        return weight_array
        
    def save_progress(self):
        # save model and scalar weights
        #weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}.h5')
        self._scaler.save(os.path.join(self.model_dir, self.model_name, f'scaler_v{self.model_index}.json'))
        weight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}')
        #os.remove(weight_path)
        #self._model.save_weights(weight_path)
        
        if self.mode == 'retrain':
            with open(weight_path, 'wb') as fh:
                pickle.dump(self.model_weight, fh)
            # save retrain history
            with open(self.hist_path, 'w') as fh:
                json.dump(self.hist, fh)
                
            # save training/validation set
            with open(os.path.join(self.model_dir, self.model_name, f'new_training_set{self.model_index}.npy'), 'wb') as fh:
                np.save(fh, self.coord)
                np.save(fh, self.energy)
                np.save(fh, self.force)
                np.save(fh, self.i_train)
                np.save(fh, self.i_val)
        else:
            with open(weight_path, 'wb') as fh:
                pickle.dump(self._model.get_weights(), fh)

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


    
    def update(self, weight_array):
        #num_layer = int(weight_array[0])
        #num_weight_per_layer = weight_array[1:num_layer+1].astype(int)
        #model_weight = weight_array[num_layer+1:num_layer+1+np.sum(num_weight_per_layer)]
        #scalar_weight = weight_array[num_layer+1+np.sum(num_weight_per_layer):]
        pl_weight_list = self._model.get_weights()
        
        # unpack the model weights
        weight_list = []
        for i in range(0, len(pl_weight_list)):
            s = pl_weight_list[i].shape
            n_weights = s[0] if len(s) == 1 else s[0]*s[1]
            weight_list.append(weight_array[:n_weights].reshape(s))
            weight_array = weight_array[n_weights:]
            #weight_list.append(model_weight[:num_weight_per_layer[i]].reshape(num_weight_per_layer[i], 1))
            #model_weight = model_weight[num_weight_per_layer[i]:]
        self._model.set_weights(weight_list)
        del pl_weight_list, weight_list
        gc.collect()
        
        # unpack the scalar weights
        n_state = self.hyper['model']['states']
        self._scaler.x_mean = np.array([float(weight_array[0])])
        self._scaler.x_std = np.array([float(weight_array[1])])
        self._scaler.energy_mean = weight_array[2:2+n_state].reshape(1, n_state)
        self._scaler.energy_std = weight_array[2+n_state:2+2*n_state].reshape(1, n_state)
        self._scaler.gradient_std = weight_array[2+2*n_state:].reshape(1, n_state, 1, 1)

        #scalar_path = os.path.join(self.model_dir, self.model_name, f'scaler_v{self.model_index}.json')
        #scalar_source = os.path.join(self.source, self.model_name, f'scaler_v{self.model_index}.json')
        #eight_path = os.path.join(self.model_dir, self.model_name, f'weights_v{self.model_index}.h5')
        #weight_source = os.path.join(self.source, self.model_name, f'weights_v{self.model_index}.h5')
        
        #shutil.copyfile(src=scalar_source, dst=scalar_path)
        #shutil.copyfile(src=weight_source, dst=weight_path)

        #self._model_setup()

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
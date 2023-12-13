#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:30:23 2023

@author: chen
"""
import matplotlib as mpl
import numpy as np
import tensorflow as tf
mpl.use('Agg')
import os
import json
import pickle
import sys

from pyNNsMD.nn_pes_src.device import set_gpu
from pyNNsMD.utils.callbacks import EarlyStopping, lr_lin_reduction, lr_exp_reduction, lr_step_reduction
from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.utils.loss import get_lr_metric, ScaledMeanAbsoluteError, r2_metric, ZeroEmptyLoss, MaskedScaledMeanAbsoluteError, masked_r2_metric, mask_MeanSquaredError

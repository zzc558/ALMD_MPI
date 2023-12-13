#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 06:19:36 2023

@author: chen

Function:
    Analysis feature importance.
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_variance_ratio(ratios):
    fig, ax = plt.subplots()
    component_label = [f"c{i}" for i in range(0, ratios.shape[0])]
    ax.bar(component_label, ratios)
    plt.show()
    plt.close()

if __name__ == "__main__":
    old_feature_path = "results/features_analysis/feature_olddata.npy"
    with open(old_feature_path, 'rb') as fh:
        feature_old = np.load(fh)
    
    pca = PCA(n_components=0.95, svd_solver='full')
    feature_pca = pca.fit_transform(feature_old)
    
    plot_variance_ratio(pca.explained_variance_ratio_)
    components = np.abs(pca.components_)
    feature_id_sorted_old = np.argsort(components, axis=1)
    
    new_feature_path = "results/features_analysis/feature_newdata.npy"
    with open(new_feature_path, 'rb') as fh:
        feature_new = np.load(fh)
    
    pca = PCA(n_components=0.95, svd_solver='full')
    feature_pca = pca.fit_transform(feature_new)
    
    plot_variance_ratio(pca.explained_variance_ratio_)
    components = np.abs(pca.components_)
    feature_id_sorted_new = np.argsort(components, axis=1)
    
    feature_importance_change = np.zeros((feature_new.shape[1],), dtype=int)
    feature_id = np.arange(0, 703, dtype=int)
    
    for i in range(0, min(feature_id_sorted_old.shape[0], feature_id_sorted_new.shape[0])):
        feature_importance_change += np.abs(np.searchsorted(feature_id_sorted_new[i], feature_id) - np.searchsorted(feature_id_sorted_old[i], feature_id))
        
    print(feature_importance_change)
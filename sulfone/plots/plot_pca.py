#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:52:47 2023

@author: chen
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def plot_pca_components(features, n_component):
    pca = PCA(n_components=n_component, svd_solver='full')
    feature_pca = pca.fit_transform(features)
    
    # Standardize the data
    #scaler = StandardScaler()
    #scaled_data = scaler.fit_transform(pca.components_)
    
    plt.imshow(pca.components_, cmap='viridis', aspect='auto', vmin=-6, vmax=6)
    plt.colorbar()
    
    plt.title('Heatmap of principal components')
    plt.xlabel('Feature')
    plt.ylabel('Component')
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    old_feature_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/results/features_analysis/feature_olddata.npy"
    with open(old_feature_path, 'rb') as fh:
        feature_old = np.load(fh)
        
    new_feature_path = "/home/chen/Documents/blueOLED/NNsForMD/sulfone/results/features_analysis/feature_newdata.npy"
    with open(new_feature_path, 'rb') as fh:
        feature_new = np.load(fh)

    for i in range(0, 10):
        plot_pca_components(feature_new[:70815+i*1500], 80)
        
    plot_pca_components(feature_new, 80)
    
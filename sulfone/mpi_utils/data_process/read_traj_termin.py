#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:30:18 2023

@author: chen
"""

import numpy as np
import os


if __name__ == "__main__":
    termin_dir = "./results/AlRun_1ns_nvt_newscript_run1/molecular_dynamic/trajData/read"
    
    normal = 0
    bond = 0
    std_high = 0
    
    for f in os.listdir(termin_dir):
        if not f.endswith(".mpi"):
            continue
        with open(os.path.join(termin_dir, f), "rb") as fh:
            termin_reasons = np.load(fh)
        if not len(termin_reasons) == 0:
            for t in termin_reasons:
                if t == 4:
                    std_high += 1
                elif t == 1:
                    bond += 1
                else:
                    normal += 1
        else:
            normal += 1
                
    print(f"Number of trajectories terminated normally: {normal}.")
    print(f"Number of trajectories terminated because of bond breaking events: {bond}.")
    print(f"Number of trajectories terminated because of high STD events: {std_high}.")
        
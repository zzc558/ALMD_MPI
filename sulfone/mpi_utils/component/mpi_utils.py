#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:50:58 2023

@author: chen
"""

import numpy as np
from mpi4py import MPI
import gc

def query_fn(status):
    print("Query function is called...")
    status.source = MPI.UNDEFINED
    status.tag = MPI.UNDEFINED
    status.cancelled = False
    status.Set_elements(MPI.BYTE, 0)
    return MPI.SUCCESS

def free_fn():
    print("Free function is called...")
    return MPI.SUCCESS

def cancel_fn(completed):
    print(f'Cancel function is called with completed = {completed}')
    return MPI.SUCCESS

def save_time_data(greq, time_dict: dict, time_keys: list, comm, fname: str, amode):
    """
    Organize the time data as numpy array to write to disk.

    Parameters
    ----------
    time_dict : dict
        dictionary containing time data.
    time_keys : list
        keys of dictionary specifying the order of time items to write to disk.

    Returns
    -------
    None
    """
    data_save = generate_time_data(time_dict, time_keys)

    fh = MPI.File.Open(comm, fname, amode)
    displacement = fh.Get_size()    # number of bytes to be skipped from the start of the file
    etype=MPI.DOUBLE    # basic unit of data access
    filetype=None     # specifies which portion of the file is visible to the process
    # MPI noncontiguous and collective writing
    fh.Set_view(displacement, etype, filetype)
    fh.Write_ordered([data_save, MPI.DOUBLE])
    fh.Close()
    
    # free memory
    del data_save
    gc.collect()
    
    return greq.Complete()

def generate_time_data(time_dict: dict, time_keys: list):
    """
    Organize the time data as numpy array to write to disk.

    Parameters
    ----------
    time_dict : dict
        dictionary containing time data.
    time_keys : list
        keys of dictionary specifying the order of time items to write to disk.

    Returns
    -------
    time_save : numpy.ndarray

    """
    time_save = np.array([], dtype=float)
    for k in time_keys:
        time_save = np.append(time_save, time_dict[k], axis=0)
        time_save = np.append(time_save, [-1,], axis=0)
        time_dict[k] = []
    
    return time_save

def save_np(traj_data: dict or list, step_keys: list, traj_keys: list, save_path: str, mode: str):
    """
    Save data as numpy array to save_path. Use threading to save time.

    Parameters
    ----------
    traj_data : dict or list
        dictionary with trajectory data. Data saved in the dictionary is moved to data_save for saving.
    step_keys : list
        list of keys of traj_data to data related to each time step (e.g. coordinates, energies, forces).
    traj_keys : list
        list of keys of traj_data to data related to trajectories (e.g. termination reason).
    save_path : str
        path to the saved data.
    mode : str
        wb for creating the file or overwrite the existing file. ab for appending on the existing file.

    Returns
    -------
    None.

    """
    if type(traj_data) == dict:
        data_save = generate_save_data(traj_data, step_keys, traj_keys)
        with open(save_path, mode) as fh:
            np.save(fh, data_save)
        del data_save
        gc.collect()
    
    elif type(traj_data) == list:
        with open(save_path, mode) as fh:
            for d in traj_data:
                np.save(fh, d)

def generate_save_data(traj_data: dict, step_keys: list, traj_keys: list):
    """
    Organize trajectory data for MPI writing to the disk.
    
    Parameters
    ----------
    traj_data : dict
        dictionary with trajectory data. Data saved in the dictionary is moved to data_save for saving.
    step_keys : list
        list of keys of traj_data to data related to each time step (e.g. coordinates, energies, forces).
    traj_keys : list
        list of keys of traj_data to data related to trajectories (e.g. termination reason).

    Returns
    -------
    data_save: numpy.ndarray
        Numpy array containing trajectory data for MPI writing to the disk.
    """
    data_save = np.array([], dtype=float)
    # add step data of completed trajectory
    for k in step_keys:
        for traj in traj_data[k][:-1]:
            data_save = np.concatenate((data_save, np.array(traj, dtype=float).flatten(), [-1.0,]), axis=0)
        data_save = np.append(data_save, [-2.0,], axis=0)
        del traj_data[k][:-1]
        gc.collect()
    data_save = np.append(data_save, [-3.0,], axis=0)
    
    # add trajectory data of completed trajectory
    for k in traj_keys:
        data_save = np.concatenate((data_save, np.array(traj_data[k], dtype=float).flatten(), [-1.0,]), axis=0)
        del traj_data[k]
        gc.collect()
        traj_data[k] = []
    data_save = np.append(data_save, [-4.0,], axis=0)
    
    # add data of the trajectory that is not completed
    for k in step_keys:
        for traj in traj_data[k][-1]:
            data_save = np.concatenate((data_save, np.array(traj, dtype=float).flatten(), [-1.0,]), axis=0)
        data_save = np.append(data_save, [-2.0,], axis=0)
        del traj_data[k]
        gc.collect()
        traj_data[k] = [[],]
    data_save = np.append(data_save, [-5.0,], axis=0)
    
    return data_save
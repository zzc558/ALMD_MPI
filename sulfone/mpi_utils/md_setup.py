#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:14:13 2022

@author: Chen Zhou
"""
import numpy as np
import random
import sys
import json
import os
import time

#from pyrai2md import PYRAI2MD
from PyRAI2MD.variables import ReadInput
from PyRAI2MD.Utils.sampling import Element
from PyRAI2MD.Utils.coordinates import ReadInitcond, PrintCoord
from PyRAI2MD.Molecule.trajectory import Trajectory
from PyRAI2MD.methods import QM
from PyRAI2MD.Dynamics.aimd import AIMD


#class MDwithNN(PYRAI2MD):

def load_input(input_file):
    with open(input_file, 'r') as file:
        try:
            input_dict = json.load(file)

        except:
            with open(input_file, 'r') as file:
                input_dict = file.read().split('&')

    return input_dict

def Sampling(title, nesmb, iseed, temp, dist, method, init_cond_path):
    ## This function recieves input information and does sampling
    ## This function use Readdata to call different functions toextract vibrational frequency and mode
    ## This function calls Boltzmann or Wigner to do sampling
    ## This function returns a list of initial condition 
    ## Import this function for external usage
    
    if iseed != -1:
        random.seed(iseed)
        
    ensemble = read_inicond(nesmb, init_cond_path)
    q = open('%s-%s-%s.xyz' % (dist, title, temp),'wb')
    p = open('%s-%s-%s.velocity' % (dist, title, temp),'wb')
    pq = open('%s.init' % (title),'wb')
    m = 0
    for mol in ensemble:
        m += 1
        geom = mol[:, 0:4]
        velo = mol[:, 4:7]
        natom = len(geom)
        np.savetxt(
            q,
            geom,
            header = '%s\n [Angstrom]' % (len(geom)),
            comments='',
            fmt = '%-5s%30s%30s%30s')
        np.savetxt(
            p,
            velo,
            header = '%d [Bohr / time_au]' % (m),
            comments='',
            fmt = '%30s%30s%30s')
        np.savetxt(pq,
            mol,
            header = 'Init %5d %5s %12s%30s%30s%30s%30s%30s%22s%6s' % (
                m,
                natom,
                'X(A)',
                'Y(A)',
                'Z(A)',
                'VX(au)',
                'VY(au)',
                'VZ(au)',
                'g/mol',
                'e'),
            comments = '',
            fmt = '%-5s%30s%30s%30s%30s%30s%30s%16s%6s')
    q.close()
    p.close()
    pq.close()
    return ensemble

def read_inicond(nesmb, init_cond_path):
    atom_list = ['S', 'O', 'O', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 
                 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 
                 'H', 'H', 'C', 'H', 'H', 'H']
    bohr_to_angstrom = 0.529177249     # 1 Bohr  = 0.529177249 Angstrom

    with open(init_cond_path, 'rb') as f:
        init_coords = np.load(f)
        init_velc = np.load(f)
    
    amass = []
    achrg = []
    natom = len(atom_list)
    for a in atom_list:
        amass.append(Element(a).getMass())
        achrg.append(Element(a).getNuc())
    
    atoms = np.array(atom_list)
    amass = np.array(amass)
    amass = amass.reshape((natom, 1))
    achrg = np.array(achrg)
    achrg = achrg.reshape((natom, 1))
    
    ensemble = [] # a list of sampled  molecules, ready to transfer to external module or print out
    for s in range(0, nesmb):
        inicond = np.concatenate((init_coords[s] * bohr_to_angstrom, init_velc[s]), axis=1)
        inicond = np.concatenate((atoms.reshape(-1, 1), inicond), axis=1)
        inicond = np.concatenate((inicond, amass[:, 0: 1]), axis = 1)
        inicond = np.concatenate((inicond, achrg), axis = 1)
        ensemble.append(inicond)
        sys.stdout.write('Progress: %.2f%%\r' % ((s + 1) * 100 / nesmb))
        del inicond
    return ensemble

if __name__ == "__main__":
    start_time = time.time()
    
    input_file = sys.argv[1]
    input_dict = load_input(input_file)
    
    job_keywords = ReadInput(input_dict)

    #init_cond_path = "/home/chen/Documents/BlueOLED/BlueOledData/data/initial_conditions.npy"
    work_dir = 'exp_S6_traj5_100_fssh_off_E06'
    init_cond_path = "/home/chen/Documents/BlueOLED/BlueOledData/data/initial_condition_Traj_from_S6_Traj5.npy"
    metadata_dir = "/home/chen/Documents/BlueOLED/NNsForMD/blueOLED/pyrai2md/metadata"
    terminal_file = os.path.join(metadata_dir, "end_ml") # file indicating the end of model running
    temp = 298
    title = 'test'
    version='2.1 alpha'
    qm = job_keywords['control']['qm']
    job_keywords['version'] = version
    
    
    os.system('mkdir ' + work_dir)
    work_dir = os.path.abspath(work_dir)
    os.chdir(work_dir)
    
    mol = Sampling(title='test', nesmb=1, iseed=0, temp=temp, dist='Nothing', method='readExist', init_cond_path=init_cond_path)[-1]
    atoms, xyz, velo = ReadInitcond(mol)
    
    for it in range(31, 100):
        os.system('mkdir iter_' + str(it))
        os.chdir(os.path.join(work_dir, 'iter_' + str(it)))
        
        xyz_out = []
        for i in range(0, xyz.shape[0]):
            xyz_out.append([str(atoms[i][0]),] + xyz[i].tolist())
    
        with open('tmp', 'w') as fh:
            fh.write(str(xyz_out))
    
        initxyz_info = '%d\n%s\n%s' % (
            len(xyz),
            '%s sampled geom %s at %s K' % ('Nothing', 10, temp),
            PrintCoord(xyz_out))
    
        with open('%s.xyz' % (title), 'w') as initxyz:
            initxyz.write(initxyz_info)

        with open('%s.velo' % (title), 'w') as initvelo:
            np.savetxt(initvelo, velo, fmt='%30s%30s%30s')
        
        traj = Trajectory(mol, keywords = job_keywords)
        method = QM(qm, keywords = job_keywords, id=None)
        method.load()
        aimd = AIMD(trajectory = traj,
                    keywords = job_keywords,
                    qm = method,
                    id = None,
                    dir = None)
        aimd.run()
        os.chdir(work_dir)
        
    fh = open(terminal_file, 'w')
    fh.close()
    
    end_time = time.time()
    print(end_time - start_time)
    with open('run_time', 'w') as fh:
        fh.write(str(end_time - start_time))
    

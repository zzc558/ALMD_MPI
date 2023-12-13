#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 03:47:53 2023

@author: chen
"""

import numpy as np
import random
import sys
import json
import os
import uuid
import shutil

#from pyrai2md import PYRAI2MD
from PyRAI2MD.variables import ReadInput
from PyRAI2MD.Utils.sampling import Element
from PyRAI2MD.Utils.coordinates import ReadInitcond, PrintCoord
from PyRAI2MD.Molecule.trajectory import Trajectory
from PyRAI2MD.Dynamics.aimd import AIMD


class MDwithNN(object):
    def __init__(self, comm, global_setting, md_setting, input_file, init_cond, time_md, traj_data, rank):
        self.atom_list = global_setting['elements']
        self.input_file = input_file
        self.init_cond = init_cond
        self.job_keywords = ReadInput(self.load_input())
        self.title = 'test'
        self.version='2.1 alpha'
        self.job_keywords['version'] = self.version
        self.work_dir = os.getcwd()
        self.temp = self.job_keywords["md"]["temp"]
        self.job_keywords['md']['root'] = int(self.init_cond[-1])
        #self.traj_dir = os.path.join(self.work_dir, f'iter_{self.iteration}')
        self.traj_dir = os.path.join(self.work_dir, f'traj_{rank}')
        if not os.path.exists(self.traj_dir):
            os.makedirs(self.traj_dir)
        os.chdir(self.traj_dir)
        
        self.mol = self.Sampling(self.title, nesmb=1, iseed=0, temp=self.temp, dist="None", \
                                 method="readExist")[-1]
            
        atoms, xyz, velo = ReadInitcond(self.mol)
        
        xyz_out = []
        for i in range(0, xyz.shape[0]):
            xyz_out.append([str(atoms[i][0]),] + xyz[i].tolist())
    
        with open('tmp', 'w') as fh:
            fh.write(str(xyz_out))
    
        initxyz_info = '%d\n%s\n%s' % (
            len(xyz),
            '%s sampled geom %s at %s K' % ('Nothing', 10, self.temp),
            PrintCoord(xyz_out))
    
        with open('%s.xyz' % (self.title), 'w') as initxyz:
            initxyz.write(initxyz_info)

        with open('%s.velo' % (self.title), 'w') as initvelo:
            np.savetxt(initvelo, velo, fmt='%30s%30s%30s')
        
        bond_index = md_setting['bond_index']
        bond_limit = md_setting['bond_limit']
        traj = Trajectory(self.mol, keywords = self.job_keywords)
        method = None
        aimd = AIMD(trajectory = traj,
                    keywords = self.job_keywords,
                    qm = method,
                    id = None,
                    dir = None,
                    comm=comm,
                    bond_index=bond_index,
                    bond_limit=bond_limit,
                    time_md=time_md,
                    traj_data=traj_data)
        aimd.run()
        os.chdir(self.work_dir)
        shutil.rmtree(self.traj_dir)
    
    def load_input(self):
        with open(self.input_file, 'r') as file:
            try:
                input_dict = json.load(file)

            except:
                with open(self.input_file, 'r') as file:
                    input_dict = file.read().split('&')

        return input_dict
    
    def Sampling(self, title, nesmb, iseed, temp, dist, method):
        ## This function recieves input information and does sampling
        ## This function use Readdata to call different functions toextract vibrational frequency and mode
        ## This function calls Boltzmann or Wigner to do sampling
        ## This function returns a list of initial condition 
        ## Import this function for external usage
        
        if iseed != -1:
            random.seed(iseed)
            
        ensemble = self.read_inicond(nesmb)
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
    
    def read_inicond(self, nesmb):
        bohr_to_angstrom = 0.529177249     # 1 Bohr  = 0.529177249 Angstrom

        #with open(self.init_cond_path, 'rb') as f:
        #    init_coords = np.load(f)
        #    init_velc = np.load(f)
        
        init_coords, init_velc, root = self.init_cond
        
        amass = []
        achrg = []
        natom = len(self.atom_list)
        for a in self.atom_list:
            amass.append(Element(a).getMass())
            achrg.append(Element(a).getNuc())
        
        atoms = np.array(self.atom_list)
        amass = np.array(amass)
        amass = amass.reshape((natom, 1))
        achrg = np.array(achrg)
        achrg = achrg.reshape((natom, 1))
        
        ensemble = [] # a list of sampled  molecules, ready to transfer to external module or print out
        for s in range(0, nesmb):
            inicond = np.concatenate((init_coords[s], init_velc[s]), axis=1)
            inicond = np.concatenate((atoms.reshape(-1, 1), inicond), axis=1)
            inicond = np.concatenate((inicond, amass[:, 0: 1]), axis = 1)
            inicond = np.concatenate((inicond, achrg), axis = 1)
            ensemble.append(inicond)
            sys.stdout.write('Progress: %.2f%%\r' % ((s + 1) * 100 / nesmb))
            del inicond
        return ensemble

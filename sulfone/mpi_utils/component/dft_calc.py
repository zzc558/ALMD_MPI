#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:51:21 2023

@author: chen
"""
import json
import uuid
import os
import re
from multiprocessing.pool import Pool
import resource
import pickle
import numpy as np
from mpi4py import MPI
from argparse import ArgumentParser

from dft_utils import setulimit, ExecuteDefineString, AddStatementToControl
from xtb_utils import exportXYZ

parser = ArgumentParser()
parser.add_argument("dir", type=str)
args = parser.parse_args()


def getTMEnergies(moldir):
    eigerfile=open("%s/eiger.out"%(moldir),"r")
    eigerlines=eigerfile.readlines()
    eigerfile.close()
    total_energy=0.0
    energy_homo=0.0
    energy_lumo=0.0
    for eigerline in eigerlines:
        if len(eigerline.split())!=0:
            if eigerline.split()[0]=="Total":
                total_energy=eigerline.split()[3]
            elif eigerline.split()[0]=="HOMO:":
                energy_homo=eigerline.split()[7]
            elif eigerline.split()[0]=="LUMO:":
                energy_lumo=eigerline.split()[7]
                break
    return([float(energy_homo),float(energy_lumo),float(total_energy)])

def sci_to_float(s):
    """convert string with scientific notation to float"""
    l = re.split('[a-zA-Z]', s)
    if len(l) == 1:
        return float(l[0])
    elif len(l) == 2:
        return float(l[0]) * (10.0**int(l[1]))
    else:
        return None

def prep_coord(moldir, coords):
    coordfile = open("%s/coord"%(moldir), 'w')
    coordfile.write("$coord \n")
    for atom in coords:
        coordfile.write("   %s      %s      %s      %s\n" % (atom[1], atom[2], atom[3], atom[0]))
    coordfile.write("$user-defined bonds\n")
    coordfile.write("$end\n")
    coordfile.close()
    
def prep_control(n):    
    try:
        from StringIO import StringIO as mStringIO
    except ImportError:
        from io import StringIO as mStringIO
        
    outfile = mStringIO()
    outfile.write(f'\nTBSO\na coord\n*\nno\nb all def2-SV(P)\n*\neht\n\n\n\nscf\nconv\n6\niter\n1800\ndamp\n0.700\n\n0.050\n\nex\nrpas\n*\na {n}\n*\nrpacor 2300\n*\ny\ndft\nfunc b3-lyp\non\n*\n*\n')
    returnstring = outfile.getvalue()
    outfile.close()
    return returnstring
    
def run_dft(coords, elements, grad=False, n_ex=4, current_state=0, n_state_total=7, identifier=None):
    
    # make temporary directory for each dft calculation
    #rundir="tmp/dft_tmpdir_%s"%(uuid.uuid4())
    rundir = os.path.join(args.dir, str(uuid.uuid4()))
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    else:
        if len(os.listdir(rundir))>0:
            os.system("rm %s/*"%(rundir))
    startdir=os.getcwd()
    os.chdir(rundir)
    
    # create coord input files containing coordiantes and element for atoms
    ## create xyz file
    exportXYZ(coords, elements, 'geom.xyz')
    ## create coord from xyz
    os.system("x2t geom.xyz > coord")
    
    # create control file
    instring = prep_control(n_ex)
    ExecuteDefineString(instring)
    if identifier != None:
        scratchdir = os.path.join(args.dir, f"_{identifier}")
        os.makedirs(scratchdir)
        s_add = f"$scratch files\n    dscf  dens  {scratchdir}/dens{identifier}\n    dscf  fock  {scratchdir}/fock{identifier}\n    dscf  dfock  {scratchdir}/dfock{identifier}\n    dscf  ddens  {scratchdir}/ddens{identifier}\n    dscf  statistics  {scratchdir}/statistics{identifier}\n    dscf  errvec  {scratchdir}/errvec{identifier}\n    dscf  oldfock  {scratchdir}/oldfock{identifier}\n    dscf  oneint  {scratchdir}/oneint{identifier}"
        AddStatementToControl("control", s_add)
    
    # run dscf to calculate ground state energy
    os.system("dscf > TM.out")
    
    # check ground state results
    dscf_finished = False   
    dscf_iterations = None
    dscf_time = None
    for line in open("TM.out","r"):
        if "convergence criteria satisfied after" in line:
            dscf_iterations = int(line.split()[4]) # number of iterations to converge
        if "all done" in line:
            dscf_finished = True
            break
        if "total wall-time" in line:
            dscf_time = int(line.split()[3]) * 60 + int(line.split()[6]) # calculation time in seconds
    if dscf_iterations!=None:
        print("   --- dscf converged after %i iterations"%(dscf_iterations))
    else:
        pass
    
    energy = [0.0] * n_state_total
    if dscf_finished:
        os.system("eiger > eiger.out")
        energy[0] = getTMEnergies(".")[-1] # read ground state energy from eiger.out
    
    # run egrad to calculate exicted state energy and gradient
    grad_finished = False
    grad_iterations = 0
    grad_time = None
    if grad and dscf_finished:
        if current_state == 0:
            os.system("grad > grad.out")
            with open('grad.out', 'r') as fh:
                content = fh.readlines()
            for i in range(0, len(content)):
                if "all done" in content[i]:
                    grad_finished = True
        else:
            AddStatementToControl("control", f"$exopt  {current_state}")
            os.system("egrad > egrad.out")
            with open("egrad.out", "r") as fh:
                content = fh.readlines()
            for i in range(0, len(content)):
                if "converged!" in content[i]:
                    grad_iterations = max(grad_iterations, int(content[i-3].split()[0])) # number of iterations to converge
                if "all done" in content[i]:
                    grad_finished = True
                if "total wall-time" in content[i]:
                    grad_time = int(content[i].split()[3]) * 60 + int(content[i].split()[6]) # calculation time in seconds
                if "singlet a excitation" in content[i]:
                    energy[int(content[i].split()[0])] = float(content[i+3].split()[-1])
            if grad_iterations != None:
                print("   --- egrad converged after %i iterations"%(grad_iterations))
                
    if grad_finished and current_state > 0:
        with open("gradient", "r") as fh:
            content = fh.readlines()
        n = len(elements)
        gradient = []
        for line in content[-1-n:-1]:
            gradient.append([sci_to_float(line.split()[i]) for i in range(0, 3)])
    elif grad_finished and current_state == 0:
        if not os.path.exists("gradient"):
            gradient = None
        gradient = []
        for line in open("gradient","r"):
            if len(line.split())==3 and "grad" not in line:
                line = line.replace("D","E")
                gradient.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])
        if len(gradient)==0:
            gradient=None
    
    results = {
        'dscf_finished': dscf_finished,
        'dscf_iterations': dscf_iterations,
        'dscf_time': dscf_time,
        'grad_finished': grad_finished,
        'grad_iterations': grad_iterations,
        'grad_time': grad_time,
        'energy': energy,
        'gradient': gradient,
        'current_n':current_state
        }
    
    # back to main directory and remove temporary directory
    os.chdir(startdir)
    os.system("rm -r %s"%(rundir))
    os.system("rm -r %s"%(scratchdir))
    
    return results
    
def mult_task(identifier, data):
    coords, elements, grad, n_ex, current_state, n_state_total = data
    results = run_dft(coords, elements, grad, n_ex, current_state, n_state_total, identifier)
    return identifier, results

def adjust_order(arr, idx, l):
    out = np.empty((len(arr),), dtype=arr.dtype)
    for i in range(0, len(idx)):
        out[idx[i]*l:(idx[i]+1)*l] = arr[i*l:(i+1)*l]
    return out

def mpi_task(comm, coords, elements, engs, do_grad=False, states=None):
    rank = comm.Get_rank()
    size = comm.Get_size()
        
    eng_tmp = np.empty((int(engs.shape[0]/size), engs.shape[1]), dtype=float)
    grad_tmp = np.empty((int(len(coords)/size), len(elements), 3), dtype=float)
    scf_iter_tmp = np.empty((int(len(coords)/size),), dtype=int)
    scf_time_tmp = np.empty((int(len(coords)/size),), dtype=float)
    grad_iter_tmp = np.empty((int(len(coords)/size),), dtype=int)
    grad_time_tmp = np.empty((int(len(coords)/size),), dtype=float)
    state_tmp = np.empty((int(len(coords)/size),), dtype=int)
    idx = np.empty((int(len(coords)/size),), dtype=int)
    count = 0
    
    for i in range(rank, len(coords), size):
        n_ex = np.count_nonzero(engs[i]) - 1
        current_state = states[i] if do_grad else None
        res = run_dft(coords[i], elements, do_grad, n_ex, current_state, len(engs[i]))
        eng_tmp[count] = np.array(res['energy'], dtype=float)
        grad_tmp[count] = np.array(res['gradient'], dtype=float)
        state_tmp[count] = res['current_n']
        scf_iter_tmp[count] = res['dscf_iterations']
        scf_time_tmp[count] = res['dscf_time']
        grad_iter_tmp[count] = res['grad_iterations']
        grad_time_tmp[count] = res['grad_time']
        idx[count] = i
        count += 1
        
    if rank == 0:
        dft_energy = np.empty(engs.shape, dtype=float).flatten()
        dft_gradient = np.empty((len(coords), len(elements), 3), dtype=float).flatten()
        dscf_iterations = np.empty((len(coords),), dtype=int)
        dscf_time = np.empty((len(coords),), dtype=float)
        grad_iterations = np.empty((len(coords),), dtype=int)
        grad_time = np.empty((len(coords),), dtype=float)
        current_state = np.empty((len(coords),), dtype=int)
        idx_all = np.empty((len(coords),), dtype=int)
    else:
        dft_energy = None
        dft_gradient = None
        dscf_iterations = None
        dscf_time = None
        grad_iterations = None
        grad_time = None
        current_state = None
        idx_all = None
        
    comm.Gather([eng_tmp.flatten(), MPI.DOUBLE], dft_energy, root=0)
    del eng_tmp
    comm.Gather([grad_tmp.flatten(), MPI.DOUBLE], dft_gradient, root=0)
    del grad_tmp
    comm.Gather([state_tmp, MPI.LONG], current_state, root=0)
    del state_tmp
    comm.Gather([scf_iter_tmp, MPI.LONG], dscf_iterations, root=0)
    del scf_iter_tmp
    comm.Gather([scf_time_tmp, MPI.DOUBLE], dscf_time, root=0)
    del scf_time_tmp
    comm.Gather([grad_iter_tmp, MPI.LONG], grad_iterations, root=0)
    del grad_iter_tmp
    comm.Gather([grad_time_tmp, MPI.DOUBLE], grad_time, root=0)
    del grad_time_tmp
    comm.Gather([idx, MPI.LONG], idx_all, root=0)
    del idx
        
    dft_energy = adjust_order(dft_energy, idx_all, engs.shape[-1]).reshape(engs.shape)
    dft_gradient = adjust_order(dft_gradient, idx_all, len(elements)*3).reshape(len(coords), len(elements), 3)
    current_state = adjust_order(current_state, idx_all, 1)
    dscf_iterations = adjust_order(dscf_iterations, idx_all, 1)
    dscf_time = adjust_order(dscf_time, idx_all, 1)
    grad_iterations = adjust_order(grad_iterations, idx_all, 1)
    grad_time = adjust_order(grad_time, idx_all, 1)
    
    dft_res = {
        'dscf_iterations': dscf_iterations,
        'dscf_time': dscf_time,
        'grad_iterations': grad_iterations,
        'grad_time': grad_time,
        'energy': dft_energy,
        'gradient': dft_gradient,
        'current_state': current_state
        }
    
    return dft_res
    

if __name__ == "__main__":
    data_files = ["../data/testing_23604_random_order_correct.json", "../data/training_70815_random_order_correct.json"]
    coords = []
    engs = []
    grads = []
    for f in data_files[:1]:
        with open(f, 'r') as fh:
            v = json.load(fh)
        coords += v[0]
        engs += v[1]
        grads += v[2]
    
    assert len(coords) == len(engs) and len(coords) == len(grads)
    
    # run multiprocess jobs to check all energies and gradients
    n_workers = 76
    coords = np.array(coords)
    elements = coords[0,:,0].tolist()
    coords = np.array(coords[:,:,1:], dtype=float)
    engs=np.array(engs, dtype=float)
    current_state = []
    for i in range(0, engs.shape[0]):
        current_state.append(int(grads[i][0][0]))
    
    # parallel dft calculation with MPI
    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()
    
    #dft_res = mpi_task(comm, coords[:size*3], elements, engs, True, current_state)
    #with open('results/dft_selfrun.json', 'w') as fh:
    #    json.dump(dft_res, fh)
    
    
    
    arg = []
    for i in range(0, 11400):
    #for i in range(0, coords.shape[0]):
        n_ex = np.count_nonzero(engs[i]) - 1
        current_state = int(grads[i][0][0])
        arg.append((i, [coords[i], elements, True, n_ex, current_state, 7]))

    dft_energy = [[],] * len(coords)
    dft_gradient = [[],] * len(coords)
    dscf_iterations = [0,] * len(coords)
    dscf_time = [0.0] * len(coords)
    grad_iterations = [0,] * len(coords)
    grad_time = [0.0,] * len(coords)

    for n in range(0, 150, 15):
        # parallel dft calculations with multiprocessing
        with Pool(n_workers) as pool:
            results = pool.starmap_async(mult_task, arg[n*n_workers:(n+15)*n_workers])
            for res in results.get():
                i, res = res
                dft_energy[i] = res['energy']
                dft_gradient[i] = [res['current_n'], res['gradient']]
                dscf_iterations[i] = res['dscf_iterations']
                dscf_time[i] = res['dscf_time']
                grad_iterations[i] = res['grad_iterations']
                grad_time[i] = res['grad_time']
        
        dft_res = {
                'dscf_iterations': dscf_iterations,
                'dscf_time': dscf_time,
                'grad_iterations': grad_iterations,
                'grad_time': grad_time,
                'energy': dft_energy,
                'gradient': dft_gradient
                }
        with open('results/dft_selfrun.json', 'w') as fh:
            json.dump(dft_res, fh)
            
        
    
    
    # test case
    #n = 122
    #c = np.array(coords[n])
    #e = c[:, 0].tolist()
    #c = np.array(c[:, 1:], dtype=float)
    #eng_src = engs[n]
    #n_ex = len(eng_src) - eng_src.count(0.0) - 1
    #grad_src = grads[n]
    #current_state = int(grad_src[0][0])
    #grad_src = grad_src[0][1]
    
    #results = run_dft(c, e, True, n_ex, current_state, len(eng_src))
    #src = {
    #    'coord': c.tolist(),
    #    'element': e,
    #    'energy': eng_src,
    #    'current_state': current_state,
    #    'gradient': grad_src
    #    }
    
    #with open('results/results.json', 'w') as fh:
    #    json.dump(results, fh)
    #with open('results/source.json', 'w') as fh:
    #    json.dump(src, fh)
    
    #for i in range(0, len(engs)):
    #    if engs[i][-1] != 0.0:
    #        prep_coord('files', coords[i])
    #        c = np.array(coords[i])
    #        e = c[:,0].tolist()
    #        c = np.array(c[:, 1:], dtype=float)
    #        exportXYZ(c, e, 'files/geom.xyz')
    #        with open('files/energy_gradient', 'w') as fh:
    #            fh.write(str(engs[i]))
    #            fh.write('\n')
    #            fh.write(str(grads[i]))
        
    
        
    
    
    # extract and format list of atoms, np.ndarray of coordinates, energies, gradients
    #coords = np.array(coords)
    #atoms = coords[:,:,0].tolist()
    #coords = np.array(coords[:,:,1:], dtype=float)
    #engs=np.array(engs, dtype=float) * 27.21138624598853  # Hatree to eV

    #grad_shape = (engs.shape[0], engs.shape[1], coords.shape[1], coords.shape[2])
    #grads_all = np.zeros(grad_shape, dtype=float)
    #for i in range(0, len(grads)):
    #    for s in grads[i]:
    #        # s[0] indicate the state of gradient: 0 -> S0, 1 -> S1, ...
    #        # s[1] contains the value of gradient
    #        grads_all[i][int(s[0])] = np.array(s[1], dtype=float) * 27.21138624598853/0.52917721090380  # Hatree to eV

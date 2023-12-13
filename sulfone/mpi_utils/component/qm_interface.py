#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 02:54:25 2023

@author: chen
"""
import uuid
import os
import re
import threading
from component.xtb_utils import exportXYZ
from mpi4py import MPI

from component.dft_utils import getTMEnergies, ExecuteDefineString, AddStatementToControl

def sci_to_float(s):
    """convert string with scientific notation to float"""
    l = re.split('[a-zA-Z]', s)
    if len(l) == 1:
        return float(l[0])
    elif len(l) == 2:
        return float(l[0]) * (10.0**int(l[1]))
    else:
        return None

class QM(object):
    def __init__(self, workdir, identifier, elements, num_state_total):
        self._workdir = workdir
        self._identifier = identifier
        self._num_state_total = num_state_total
        self._elements = elements
    
    def prep_control(self, n):    
        try:
            from StringIO import StringIO as mStringIO
        except ImportError:
            from io import StringIO as mStringIO
            
        outfile = mStringIO()
        outfile.write(f'\nTBSO\na coord\n*\nno\nb all def2-SV(P)\n*\neht\n\n\n\nscf\nconv\n6\niter\n1800\ndamp\n0.700\n\n0.050\n\nex\nrpas\n*\na {n}\n*\nrpacor 2300\n*\ny\ndft\nfunc b3-lyp\non\n*\n*\n')
        returnstring = outfile.getvalue()
        outfile.close()
        return returnstring
    
    def run_dft(self, coords, grad=False, num_ex=4, current_state=0, results={}):
        
        # make temporary directory for each dft calculation
        #rundir="tmp/dft_tmpdir_%s"%(uuid.uuid4())
        rundir = os.path.join(self._workdir, str(uuid.uuid4()))
        if not os.path.exists(rundir):
            os.makedirs(rundir)
        else:
            if len(os.listdir(rundir))>0:
                os.system("rm %s/*"%(rundir))
        startdir=os.getcwd()
        os.chdir(rundir)
        
        # create coord input files containing coordiantes and element for atoms
        ## create xyz file
        exportXYZ(coords, self._elements, 'geom.xyz')
        ## create coord from xyz
        os.system("x2t geom.xyz > coord")
        
        # create control file
        instring = self.prep_control(num_ex)
        ExecuteDefineString(instring)
        if self._identifier != None:
            scratchdir = os.path.join(self._workdir, f"_{self._identifier}")
            os.makedirs(scratchdir)
            s_add = f"$scratch files\n    dscf  dens  {scratchdir}/dens{self._identifier}\n    dscf  fock  {scratchdir}/fock{self._identifier}\n    dscf  dfock  {scratchdir}/dfock{self._identifier}\n    dscf  ddens  {scratchdir}/ddens{self._identifier}\n    dscf  statistics  {scratchdir}/statistics{self._identifier}\n    dscf  errvec  {scratchdir}/errvec{self._identifier}\n    dscf  oldfock  {scratchdir}/oldfock{self._identifier}\n    dscf  oneint  {scratchdir}/oneint{self._identifier}"
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
                try:
                    dscf_time = int(line.split()[3]) * 60 + int(line.split()[6]) # calculation time in seconds
                except:
                    dscf_time = int(line.split()[line.split().index('seconds')-1])
        if dscf_iterations!=None:
            print("   --- dscf converged after %i iterations"%(dscf_iterations))
        else:
            pass
        
        energy = [0.0] * self._num_state_total
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
                    content = fh.read()
                if "all done" in content:
                    grad_finished = True
                os.system("escf > escf.out")
                with open('escf.out', 'r') as fh:
                    content = fh.readlines()
                eng_finished = False
                for i in range(0, len(content)):
                    if "all done" in content[i]:
                        eng_finished = True
                    if "singlet a excitation" in content[i]:
                        energy[int(content[i].split()[0])] = float(content[i+3].split()[-1])

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
            n = len(self._elements)
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
        else:
            gradient = None
        
        results['dscf_finished'] =  dscf_finished
        results['dscf_iterations'] = dscf_iterations
        results['dscf_time'] = dscf_time
        results['grad_finished'] = grad_finished
        results['grad_iterations'] = grad_iterations
        results['grad_time'] = grad_time
        results['energy'] = energy
        results['gradient'] = gradient
        results['current_n'] = current_state

        #results = {
        #    'dscf_finished': dscf_finished,
        #    'dscf_iterations': dscf_iterations,
        #    'dscf_time': dscf_time,
        #    'grad_finished': grad_finished,
        #    'grad_iterations': grad_iterations,
        #    'grad_time': grad_time,
        #    'energy': energy,
        #    'gradient': gradient,
        #    'current_n':current_state
        #    }
        
        # back to main directory and remove temporary directory
        os.chdir(startdir)
        os.system("rm -r %s"%(rundir))
        os.system("rm -r %s"%(scratchdir))
        
        #return results
    
class QM_MPI(QM):
    def _run_dft(self, greq, **kwargs):
        super(QM_MPI, self).run_dft(**kwargs)
        greq.Complete()
    
    def _query_fn(self, status):
        print("Query function is called...")
        status.source = MPI.UNDEFINED
        status.tag = MPI.UNDEFINED
        status.cancelled = False
        status.Set_elements(MPI.BYTE, 0)
        
        return MPI.SUCCESS
    
    def _free_fn(self):
        print("Free function is called...")
        return MPI.SUCCESS
    
    def _cancel_fn(self, completed):
        print(f'Cancel function is called with completed = {completed}')
        return MPI.SUCCESS
    
    
    def mpi_dft(self, **kwargs):
        greq = MPI.Grequest.Start(self._query_fn, self._free_fn, self._cancel_fn)
        dft_thread = threading.Thread(target=self._run_dft, name=f'dft_thread_rank{self._identifier}', args=(greq,), kwargs=kwargs)
        dft_thread.daemon = True
        dft_thread.start()
        
        return greq
    
if __name__ == "__main__":
    # test the implementation of QM_MPI
    from argparse import ArgumentParser
    import json
    import time
    import numpy as np
    
    parser = ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank < size - 1:
        data_file = "../data/training_70815_random_order_correct.json"
        coords = []

        with open(data_file, 'r') as fh:
            coords = json.load(fh)[0]
        coords = np.array(coords)
        elements = coords[0,:,0].tolist()
        coords = np.array(coords[:size-1,:,1:], dtype=float)

        q = QM_MPI(args.dir, rank, elements, 7)
        greq = q.mpi_dft(coords=coords[rank], grad=True, num_ex=6, current_state=2)
        dft_status = MPI.Status()

        while True:
            finish_flag = np.empty((1,), dtype=bool)
            comm.Recv([finish_flag, MPI.BOOL], source=size-1, tag=1)
            if bool(finish_flag):
                print(f"Rank {rank} quits.")
                break
            else:
                comm.Send([np.bool_(greq.Test(dft_status)), MPI.BOOL], dest=size-1, tag=2)


    else:
        work_rank = tuple(range(0, size-1))
        work_finished = np.array([False,]*(size-1), dtype=bool)
        for i in range(0, 10):
            time.sleep(180)
            for r in range(0, len(work_rank)):
                if not work_finished[r]:
                    comm.Send([np.bool_(False), MPI.BOOL], dest=work_rank[r], tag=1)
                    f = np.empty((1,), dtype=bool)
                    comm.Recv([f, MPI.BOOL], source=work_rank[r], tag=2)
                    work_finished[r] = f[0]

                if work_finished[r]:
                    # TODO: this leads to unexpected send operation when work rank quited already. Needs to be changed.
                    comm.Send([np.bool_(True), MPI.BOOL], dest=work_rank[r], tag=1)
                    print(f'Rank {rank}: rank {work_rank[r]} finished dft calculation.')
                else:
                    print(f'Rank {rank}: rank {work_rank[r]} is still working...')

            if work_finished.all():
                print(f"Rank {rank}: all dft calculations are finished.")
                break

        if work_finished.all():
            print(f"Rank {rank}: All dft calculations finished in 30 minutes!")
        else:
            print(f"Rank {rank}: Some dft calculations doesn't finish in 30 minutes!")
        

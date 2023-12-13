#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:28:14 2023

@author: chen
"""

import numpy as np
from mpi4py import MPI
import sys, gc, threading, os, time, json, pickle, shutil

from component.al_setting import DEFAULT_AL_SETTING as setting
#from component.nn import NNforMPI, eval_std
#from component.dft_calc import run_dft
#from component.ml_interface import MLForAl, masked_MAE
from md_utils import MDwithNN
from component.qm_interface import QM_MPI
#from pyNNsMD.utils.loss import r2_metric

RANK_MG = 0                       # rank of manager process (MUST be fixed as rank 0)
HtoEv = 27.21138624598853
AToBohr = 1.889725989

def eval_std(data_array, threshold):
    std = np.std(data_array, axis=0, ddof=1)
    return np.where((std > threshold).any(axis=1))[0]

def save_np(data_save: dict):
    #assert len(fpath) == len(num_data_per_file), "Error: Number of data objects to save is not equal to number of file paths."
    for fpath, fdata in data_save.items():
        with open(fpath, 'wb') as fh:
            for d in fdata:
                np.save(fh, d)

def save_pickle(data_save: dict):
    for fpath, fdata in data_save.items():
        with open(fpath, 'ab') as fh:
            pickle.dump(fdata, fh)

def copy_dir(src, dest):
    shutil.copytree(src, dest, dirs_exist_ok=True)

if __name__ == "__main__":
    # TODO
    t_start = time.time()
    ## read settings and hyperparameters
    print(setting)
    global_setting = setting['global']
    # keep record of identifiers (ranks) of each process
    n_pl = global_setting['pl_process']    # number of passive learner processes
    n_md = global_setting['md_process']    # number of molecular dynamic processes
    n_qm = global_setting['qm_process']    # number of quantum mechanics processes
    n_ml = global_setting['ml_process']    # number of machine learning processes
    model_per_pl = global_setting['model_per_pl'] # number of PL models per PL process
    model_total = model_per_pl * n_pl    # total number of models
    retrain_step = setting['manager']['retrain_step']    # increment of training set
    tmp_dir = sys.argv[-1]               # TMP directory on cluster node for dft calculation
    #md_input = sys.argv[1]                 # input file containing MD settings
    dft_wait_time = global_setting['dft_time']    # estimated dft running time (energy+force) in seconds
    gpu_per_node = global_setting['n_gpu']
    update_time = global_setting['update_time']
    
    rank_pl = tuple(range(1, n_pl+1))      # list of ranks of passive learner processes
    rank_md = tuple(range(n_pl+1, n_md+n_pl+1))    # list of ranks of molecular dynamic processes
    rank_qm = tuple(range(n_md+n_pl+1, n_qm+n_md+n_pl+1))    # list of ranks of quantum mechanic processes
    rank_ml = tuple(range(n_qm+n_md+n_pl+1, n_ml+n_qm+n_md+n_pl+1))    # list of ranks of machine learner processes
    
    # molecule information
    elements = global_setting['elements']
    n_atoms = len(elements)
    n_states = global_setting['n_states']
    # directory to store all the scratches/results
    al_dir = global_setting['res_dir']
    directory = os.path.join(tmp_dir, al_dir)
    mg_path = os.path.join(directory, "process_manager")
    md_path = os.path.join(directory, "molecular_dynamic")
    ml_path = os.path.join(directory, "machine_learning")
    qm_path = os.path.join(directory, "quantum_mechanics")
    pl_path = os.path.join(directory, "passive_learner")
    
    ## MPI set up
    comm_world = MPI.COMM_WORLD            # communicator to pass message among all processes
    rank = comm_world.Get_rank()           # rank of current process
    size = comm_world.Get_size()           # total number of processes
    
    # make directories
    if rank == RANK_MG:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(mg_path, exist_ok=True)
        os.makedirs(md_path, exist_ok=True)
        os.makedirs(ml_path, exist_ok=True)
        os.makedirs(qm_path, exist_ok=True)
        os.makedirs(pl_path, exist_ok=True)
    t_pl_mg = 0                            # mpi tag for communication between PL and MG process
    t_md_mg = 1                            # mpi tag for communication between MD and MG process
    t_ml_mg = 2                            # mpi tag for communication between ML and MG process
    t_ml_pl = 3                            # mpi tag for communication between ML and PL process
    t_ml = 4                               # mpi tag for communication among ML processes
    t_md = 5
    t_pl = 6
    t_qm_mg = list(range(7, n_qm+7))       # mpi tag for communication between QMs and MG process
    group_world = comm_world.Get_group()
    # create communicator to pass message between PL processes and manager process
    group_pl_mg = group_world.Incl([RANK_MG,] + list(rank_pl))
    comm_pl_mg = comm_world.Create_group(group_pl_mg, tag=t_pl_mg)
    # create communicator to pass message between MD processes and manager process
    group_md_mg = group_world.Incl([RANK_MG,] + list(rank_md))
    comm_md_mg = comm_world.Create_group(group_md_mg, tag=t_md_mg)
    # create communicator to pass message between ML processes and manager process
    group_ml_mg = group_world.Incl([RANK_MG,] + list(rank_ml))
    comm_ml_mg = comm_world.Create_group(group_ml_mg, tag=t_ml_mg)
    # create communicator to pass weights between ML process and PL process
    group_ml_pl = group_world.Incl(list(rank_pl) + [rank_ml[0],])
    comm_ml_pl = comm_world.Create_group(group_ml_pl, tag=t_ml_pl)
    # create communicator to collect weights among ML processes
    group_ml = group_world.Incl(list(rank_ml))
    comm_ml = comm_world.Create_group(group_ml, tag=t_ml)
    
    # TODO: MPI I/O
    group_md = group_world.Incl(list(rank_md))
    comm_md = comm_world.Create_group(group_md, tag=t_md)
    group_pl = group_world.Incl(list(rank_pl))
    comm_pl = comm_world.Create_group(group_pl, tag=t_pl)
    data_count_save = 200
    
    assert size == len(rank_pl)+len(rank_md)+len(rank_qm)+len(rank_ml)+1,\
        "Error: number of processes not equal to size of ranks"
    stop_run = False
    
    # Passive Learner Process (PL)
    # Recive coordinates from MD through MG, make predictions and send back to MD through MG
    # Copy new model and scaler weights from ML process and update models
    if rank in rank_pl:
        from component.ml_interface import MLForAl
        # create directory to store data of PL processes
        #pl_path = os.path.join(directory, "passive_learner")
        #if not os.path.exists(pl_path):
        #    os.makedirs(pl_path)
        # each rank writes to its own log file
        while not os.path.exists(pl_path):
            time.sleep(1)
        fstdout = open(os.path.join(pl_path, f"pllog_{rank}.txt"), 'w')
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start passive learner process...")
        
        # read settings and initilize models
        pl_setting = setting['passive']
        gpu_list = pl_setting['gpu_list']
        model_name = pl_setting['model_name']
        model_path = pl_setting['path']
        test_path = pl_setting['test_path']
        model_hyper = setting['model_hyper']
        ml_source = setting['ml']['path']
        model_index = (comm_pl_mg.Get_rank() - 1) % gpu_per_node
        pl_input = {
            'gpu_index': model_index,
            'model_index': model_index,
            'model_dir': model_path,
            'model_name': 'eg',
            'hyper': model_hyper,
            'source': ml_source,
            'mode': 'prediction'
            }
        
        print(f"Rank {rank}: Initilize the model...")
        #nn = NNforMPI(gpu_list, model_path, model_per_node, model_name, model_hyper, source=ml_source, test_path=test_path, mode='prediction')
        nn = MLForAl(method='nn', kwargs=pl_input)
        
        with open(test_path, 'r') as fh:
            c, e, g = json.load(fh)
        c = np.array(c)
        c = np.array(c[:,:,1:], dtype=float)
        e = np.array(e, dtype=float) * HtoEv
        g_shape = (e.shape[0], e.shape[1], c.shape[1], c.shape[2])
        g_all = np.zeros(g_shape, dtype=float)
        for i in range(0, len(g)):
            for s in g[i]:
                # s[0] indicate the state of gradient: 0 -> S0, 1 -> S1, ...
                # s[1] contains the value of gradient
                g_all[i][int(s[0])] = np.array(s[1], dtype=float) * HtoEv * AToBohr
        pl_testset = [c, e, g_all]
        
        req_weight = None    # request used to receive model weights from ML process
        
        #TODO: running time
        ftime_pl = os.path.join(pl_path, 'pltime')
        if os.path.exists(ftime_pl):
            amode = MPI.MODE_APPEND|MPI.MODE_WRONLY
        else:
            amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
        time_pl = {
            'bcast': np.array([], dtype=float),
            'predict': np.array([], dtype=float),
            'gather': np.array([], dtype=float),
            'update': np.array([], dtype=float)
            }
        #try:
        #   with open(os.path.join(pl_path, f'pltime_{rank}.json'), 'r') as fh:
        #        time_pl = json.load(fh)
        #except:
        #    time_pl = {
        #        'bcast': [],
        #        'predict': [],
        #        'gather':[],
        #        'update': []
        #        }
        #t_start = time.time()
        save_progress = False
        pl_write_req = None
        fh_pl = MPI.File.Open(comm_pl, ftime_pl, amode)
        while not stop_run:
            #if time.time() - t_start >= update_time:
            #    save_progress = True
            #    t_start = time.time()
            #else:
            #    save_progress = False
            
            # receive ML status and coordinates through MG process
            print(f"Rank {rank}: receiving model weights from ML...")
            if req_weight == None:
                weight_array_collect = None
                weight_array = np.empty((nn.get_num_weight()), dtype=float)
                req_weight = comm_ml_pl.Iscatter([weight_array_collect, MPI.DOUBLE], [weight_array, MPI.DOUBLE], root=comm_ml_pl.Get_size()-1)
                pl_updated = np.array([0,], dtype=float)
            elif req_weight.Test():
                print()
                print("New weights arrived...")
                req_weight = None
                t4 = time.time()
                nn.update(weight_array)    # ML retraining finished, update PL models
                # free memory
                del weight_array
                gc.collect()
                print("Model updated!")
                print()
                t5 = time.time()
                #TODO: running time
                time_pl['update'] = np.append(time_pl['update'], t5-t4)
                pl_updated = np.array([1,], dtype=float)
            else:
                pl_updated = np.array([0,], dtype=float)
            #coords_to_pl = np.empty((n_md*n_atoms*3+1,), dtype=float)
            print(f"Rank {rank}: receiving coordiantes...")
            #comm_pl_mg.Bcast([coords_to_pl, MPI.DOUBLE], root=RANK_MG)
            coords_to_pl = None
            t0 = time.time()
            coords_to_pl = comm_pl_mg.bcast(coords_to_pl, root=RANK_MG)
            t1 = time.time()
            if not coords_to_pl[1] is None:
                #t4 = time.time()
                #nn.update(weight_array)    # ML retraining finished, update PL models
                ## free memory
                #del weight_array
                #gc.collect()
                #t5 = time.time()
                #time_pl['update'].append(t5-t4)
                coord_for_qm = coords_to_pl[1]
                if coord_for_qm.shape[0] != 0:
                    eng, _ = nn.predict(np.concatenate((pl_testset[0], coord_for_qm), axis=0))
                else:
                    eng, _ = nn.predict(pl_testset[0])
                test_eng_pred = eng[:pl_testset[0].shape[0]]
                qm_eng_pred = eng[pl_testset[0].shape[0]:]
                #test_eng_mae = masked_MAE(pl_testset[1], test_eng_pred)
                #test_eng_r2 = masked_r2(pl_testset[1], test_eng_pred)
                #data_to_mg = np.concatenate((np.array([float(test_eng_mae), float(test_eng_r2)]), test_eng_pred.flatten(), qm_eng_pred.flatten()), axis=0)
                data_to_mg = np.concatenate((test_eng_pred.flatten(), qm_eng_pred.flatten()), axis=0)
                print("PL models have been updated.")
            else:
                data_to_mg = np.array([], dtype=float)
                
            #coords = coords_to_pl[1:].reshape(n_md, n_atoms, 3)
            coords = coords_to_pl[0]
            # make predictions
            eng_pred, force_pred = nn.predict(coords)
            eng_pred = eng_pred[:,:n_states]
            force_pred = force_pred[:,:n_states]
            t2 = time.time()
            print(f"Rank {rank}: predict energy with shape {eng_pred.shape} and force shape {force_pred.shape}")
            print(f"Rank {rank}: send predictions back to MG process...")
            # send predictions back through MG process
            data_to_mg = np.concatenate((data_to_mg, eng_pred.flatten(), force_pred.flatten(), pl_updated.flatten()), axis=0)
            print(f"Rank {rank}: shape of data sent back to MG {data_to_mg.shape}")
            data_from_pl = None
            comm_pl_mg.Gather([data_to_mg, MPI.DOUBLE], [data_from_pl, MPI.DOUBLE], root=RANK_MG)
            print("Predictions have been sent back to MG process.")
            t3 = time.time()
            
            #TODO: running time
            time_pl['bcast'] = np.append(time_pl['bcast'], t1-t0)
            time_pl['predict'] = np.append(time_pl['predict'], t2-t1)
            time_pl['gather'] = np.append(time_pl['gather'], t3-t2)
            #if len(time_pl['bcast']) // data_count_save >= 1:
            if coords_to_pl[-1] == 1:
                save_progress = True
            if save_progress and (pl_write_req is None or pl_write_req.Test()):
                data_save = np.zeros((data_count_save*len(time_pl.keys()),), dtype=float)
                data_save[:data_count_save] = time_pl['bcast'][:data_count_save]
                data_save[data_count_save:data_count_save*2] = time_pl['predict'][:data_count_save]
                data_save[data_count_save*2:data_count_save*3] = time_pl['gather'][:data_count_save]
                data_save[data_count_save*3:data_count_save*3+len(time_pl['update'])] = time_pl['update']
                #fh = MPI.File.Open(comm_pl, ftime_pl, amode)
                fsize = fh_pl.Get_size()
                displacement = fsize + MPI.DOUBLE.Get_size()*data_save.shape[0]*(rank-rank_pl[0])
                fh_pl.Set_view(displacement)
                pl_write_req = fh_pl.Iwrite_all(data_save)
                #fh.Close()
                save_progress = False
                time_pl = {
                    'bcast': time_pl['bcast'][data_count_save:],
                    'predict': time_pl['predict'][data_count_save:],
                    'gather': time_pl['gather'][data_count_save:],
                    'update': np.array([], dtype=float)
                    }
                #nn.save_progress()
                #with open(os.path.join(pl_path, f'pltime_{rank}.json'), 'w') as fh:
                #    json.dump(time_pl, fh)
                print("model progress saved...")
        
        print("The End.")
        fh_pl.Close()
        fstdout.close()
            
    # Passive Molecular Dynamic (MD)
    # Propagate trajectories. Send coordinates to PL through MG
    if rank in rank_md:
        # create directory to store data of MD processes
        #md_path = os.path.join(directory, "molecular_dynamic")
        #if not os.path.exists(md_path):
        #    os.makedirs(md_path)
        # create working directory for each rank
        #work_dir = os.path.abspath(os.path.join(md_path, f'rank{rank}'))
        #if not os.path.exists(work_dir):
        #    os.makedirs(work_dir)
        # each rank writes to its own log file
        while not os.path.exists(md_path):
            time.sleep(1)
        fstdout = open(os.path.join(md_path, f"mdlog_{rank}.txt"), 'w')
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start molecular dynamic process...")
        
        md_setting = setting['md']
        initial_cond_path = md_setting['initial_cond_path']
        md_input = os.path.abspath(md_setting['input_file'])
        # load initial conditions
        with open(initial_cond_path, 'rb') as fh:
            init_coord = np.load(fh)
            init_velc = np.load(fh)
            root = np.load(fh)
        root += 1
        init_idx = np.array(range(0, init_coord.shape[0]), dtype=int)
        
        #try:
        #    with open(os.path.join(md_path, f'mdtime_{rank}.json'), 'r') as fh:
        #        time_md = json.load(fh)
        #except:
        #    time_md = {
        #        'gather': [],
        #        'scatter': [],
        #        'prop': []
        #        }
        #try:
        #    with open(os.path.join(md_path, f'traj_data_{rank}.json'), 'r') as fh:
        #        traj_data = json.load(fh)
        #except:
        #    traj_data = {
        #        'energy': [],
        #        'force': [],
        #        'coord': [],
        #        'state': [],
        #        }
        
        os.chdir(md_path)
        #TODO: running time
        ftime_md = os.path.join("mdtime")
        if os.path.exists(ftime_md):
            amode_time = MPI.MODE_APPEND|MPI.MODE_WRONLY
        else:
            amode_time = MPI.MODE_WRONLY|MPI.MODE_CREATE
        time_md = {
            'gather': [],
            'scatter': [],
            'prop': []
            }
        
        # To save trajectory data
        ftraj_md = os.path.join("traj_data")
        if os.path.exists(ftraj_md):
            amode_traj = MPI.MODE_APPEND|MPI.MODE_WRONLY
        else:
            amode_traj = MPI.MODE_WRONLY|MPI.MODE_CREATE
        traj_data = {
            'energy': [],
            'force': [],
            'coord': [],
            'state': [],
            'save_data': 0,
            }
            
        #os.chdir(md_path)
        t_start = time.time()
        save_time = False
        save_traj = False
        md_time_req = None
        md_traj_req = None
        data_count_save_traj = int(1.5*data_count_save)
        fh_time = MPI.File.Open(comm_md, ftime_md, amode_time)
        fh_traj = MPI.File.Open(comm_md, ftraj_md, amode_traj)
        while not stop_run:
            if time.time() - t_start >= update_time:
                save_progress = True
                t_start = time.time()
            else:
                save_progress = False
            i = np.random.choice(init_idx, size=1)
            print("MD process started...")
            for k in traj_data.keys():
                if k != 'save_data':
                    traj_data[k].append([])
            md_process = MDwithNN(comm_md_mg, global_setting, md_setting, md_input,\
                                  [init_coord[i], init_velc[i], root[i]], time_md, traj_data, rank)
            del md_process
            print("Finished trajectory.")
            #if len(time_md['gather']) // data_count_save >= 1:
            if traj_data['save_data'] > 0:
                save_time = True
            if traj_data['save_data'] > 1:
                save_traj = True
                traj_data['save_data'] = 0
            if save_time and (md_time_req is None or md_time_req.Test()) and (md_traj_req is None or md_traj_req.Test()):
                time_save = np.zeros((data_count_save*len(time_md.keys()),), dtype=float)
                time_save[:data_count_save] = time_md['gather'][:data_count_save]
                time_save[data_count_save:data_count_save*2] = time_md['scatter'][:data_count_save]
                time_save[data_count_save*2:data_count_save*3] = time_md['prop'][:data_count_save]
                #fh_time = MPI.File.Open(comm_md, ftime_md, amode_time)
                fsize = fh_time.Get_size()
                displacement = fsize + MPI.DOUBLE.Get_size()*time_save.shape[0]*(rank-rank_md[0])
                fh_time.Set_view(displacement)
                md_time_req = fh_time.Iwrite_all(time_save)
                #fh_time.Close()
                save_time = False
                time_md['gather'] = time_md['gather'][data_count_save:]
                time_md['scatter'] = time_md['scatter'][data_count_save:]
                time_md['prop'] = time_md['prop'][data_count_save:]
            #if len(traj_data['state']) // data_count_save_traj >= 1:
            #    save_traj = True
            if save_traj and (md_traj_req is None or md_traj_req.Test()) and (md_time_req is None or md_time_req.Test()):
                traj_data_write = np.append(np.array(traj_data['coord'][:data_count_save_traj], dtype=float).flatten(), np.array(traj_data['energy'][:data_count_save_traj], dtype=float).flatten(), axis=0)
                traj_data_write = np.append(traj_data_write, np.array(traj_data['state'][:data_count_save_traj], dtype=float).flatten(), axis=0)
                traj_data_write = np.append(traj_data_write, np.array(traj_data['force'][:data_count_save_traj], dtype=float).flatten(), axis=0)
                #fh_traj = MPI.File.Open(comm_md, ftraj_md, amode_traj)
                fsize = fh_traj.Get_size()
                displacement = fsize + MPI.DOUBLE.Get_size()*traj_data_write.shape[0]*(rank-rank_md[0])
                fh_traj.Set_view(displacement)
                md_traj_req = fh_traj.Iwrite_all(traj_data_write)
                #fh_traj.Close()
                save_traj = False
                traj_data['coord'] = traj_data['coord'][data_count_save_traj:]
                traj_data['energy'] = traj_data['energy'][data_count_save_traj:]
                traj_data['state'] = traj_data['state'][data_count_save_traj:]
                traj_data['force'] = traj_data['force'][data_count_save_traj:]
            #if save_progress:
            #    with open(os.path.join(md_path, f'mdtime_{rank}.json'), 'w') as fh:
            #        json.dump(time_md, fh)
            #    with open(os.path.join(md_path, f'traj_data_{rank}.json'), 'w') as fh:
            #        json.dump(traj_data, fh)
        
        print("The End.")
        fh_time.Close()
        fh_traj.Close()
        fstdout.close()
            
    # Quantum Mechanics process (QM)
    # Receive geometries from MG and run DFT calculations
    if rank in rank_qm:
        # create directory to store data of QM processes
        #qm_path = os.path.join(directory, "quantum_mechanics")
        #if not os.path.exists(qm_path):
        #    os.makedirs(qm_path)
        
        # each rank writes to its own log file
        while not os.path.exists(qm_path):
            time.sleep(1)
        fstdout = open(os.path.join(qm_path, f"qmlog_{rank}.txt"), 'w')
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start molecular dynamic process...")
        
        tag_here = t_qm_mg[rank-rank_qm[0]]    # MPI tag for this QM process
        qm = QM_MPI(tmp_dir, rank, elements, n_states)
        
        #try:
        #    with open(os.path.join(qm_path, f'qmtime_{rank}.json'), 'r') as fh:
        #        time_qm = json.load(fh)
        #except:
        #    time_qm = {
        #        'recv': [],
        #        'send':[]
        #        }
        
        t_start = time.time()
        while not stop_run:
            #if time.time() - t_start >= update_time:
            #    save_progress = True
            #    t_start = time.time()
            #else:
            #    save_progress = False
            # receive coordinates from MG process
            coord_from_mg = np.empty((1+n_atoms*3,), dtype=float)
            t0 = time.time()
            comm_world.Recv([coord_from_mg, MPI.DOUBLE], source=RANK_MG, tag=tag_here)
            t1 = time.time()
            current_state = int(coord_from_mg[0])
            coord_from_mg = coord_from_mg[1:].reshape(n_atoms, 3)
            print("Received coordinates from MG process for DFT calculation.")
            # run dft calculations
            results = {}
            greq = qm.mpi_dft(coords=coord_from_mg, grad=True, num_ex=n_states-1, current_state=current_state, results=results)
            s = MPI.Status()
            time.sleep(dft_wait_time)
            while not greq.Test(s):
                comm_world.Send([np.zeros((1,), dtype=int), MPI.LONG], dest=RANK_MG, tag=tag_here)
                time.sleep(120)
            comm_world.Send([np.ones((1,), dtype=int), MPI.LONG], dest=RANK_MG, tag=tag_here)
            #results = run_dft(coord_from_mg, elements, grad=True, n_ex=n_states-1, current_state=current_state, n_state_total=n_states, identifier=rank)
            dft_energy = np.array(results['energy'], dtype=float)
            assert current_state == results['current_n'], "Error: check the current state sent to dft calculation and received."
            dft_force = np.zeros((n_states, n_atoms, 3), dtype=float)
            dft_force[current_state] = np.array(results['gradient'], dtype=float)
            print("Done with the DFT calculation.")
            # send energy and force back to MG process
            t2 = time.time()
            comm_world.Send([np.concatenate((coord_from_mg.flatten(), dft_energy.flatten(), dft_force.flatten()),axis=0), MPI.DOUBLE], dest=RANK_MG, tag=tag_here)
            t3 = time.time()
            #time_qm['recv'].append(t1-t0)
            #time_qm['send'].append(t3-t2)
            #if save_progress:
            #    with open(os.path.join(qm_path, f'qmtime_{rank}.json'), 'w') as fh:
            #        json.dump(time_qm, fh)
            print("Data have been send back to MG process.")
        
        print("The End.")
        fstdout.close()
        
    # Machine learning Process (ML)
    # Receive coordinates, energy and force from MG
    # Retrain the model
    # Notify MG once finishing retraining
    if rank in rank_ml:
        from component.ml_interface import MLForAl
        # create directory to store data of ML processes
        #ml_path = os.path.join(directory, "machine_learning")
        #if not os.path.exists(ml_path):
        #    os.makedirs(ml_path)
        # each rank writes to its own log file
        while not os.path.exists(ml_path):
            time.sleep(1)
        fstdout = open(os.path.join(ml_path, f"mllog_{rank}.txt"), 'w')
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start machine learning process...")
        
        # read settings and initilize models
        pl_setting = setting['passive']
        gpu_list = pl_setting['gpu_list']
        model_name = pl_setting['model_name']
        model_hyper = setting['model_hyper']
        model_path = setting['ml']['path']
        model_index = (comm_ml_mg.Get_rank() - 1) % n_ml
        req_weight = None    # request object used to check if weights have reached PL processes
        
        #try:
        #    with open(os.path.join(ml_path, f'mltime_{rank}.json'), 'r') as fh:
        #        time_ml = json.load(fh)
        #except:
        #    time_ml = {
        #        "gather_weight": [],
        #        "scatter_weight": []
        #        }
        
        ml_input = {
            'gpu_index': model_index,
            'model_index': model_index,
            'model_dir': model_path,
            'model_name': 'eg',
            'hyper': model_hyper,
            'source': None,
            'mode': 'retrain'
            }
        
        print(f"Rank {rank}: Initilize the model...")
        #nn = NNforMPI(gpu_list, model_path, model_per_node, model_name, model_hyper, source=None, test_path=None, mode='retrain')
        nn = MLForAl(method='nn', kwargs=ml_input)
        
        # wait for the first set of retraining data before start retraining
        data_send = np.empty((retrain_step*(n_atoms*3+n_states+n_states*n_atoms*3),), dtype=float)
        req_ml_data = comm_ml_mg.Ibcast([data_send, MPI.DOUBLE], root=RANK_MG)
        req_ml_data.Wait()
        new_coord = data_send[:retrain_step*n_atoms*3].reshape(retrain_step, n_atoms, 3)
        data_send = data_send[retrain_step*n_atoms*3:]
        new_energy = data_send[:retrain_step*n_states].reshape(retrain_step, n_states) * HtoEv
        new_force = data_send[retrain_step*n_states:].reshape(retrain_step, n_states, n_atoms, 3) * HtoEv * AToBohr
        nn.add_trainingset(new_coord, new_energy, new_force)
        t_start = time.time()
        save_progress = False
        while not stop_run:
            if not save_progress:
                if time.time() - t_start >= update_time:
                    save_progress = True
            # receive coordinates, energy and force from MG process
            data_send = np.empty((retrain_step*(n_atoms*3+n_states+n_states*n_atoms*3),), dtype=float)
            req_ml_data = comm_ml_mg.Ibcast([data_send, MPI.DOUBLE], root=RANK_MG)
            # starting retraining with current training set
            # stop retraining when new training data arrive
            nn.retrain(req_ml_data)
            # when retraining finished or early stopping
            # wait until new retraining data arrive
            req_ml_data.Wait()
            print("Retraining finished.")
            t0 = time.time()
            # get weight array
            weight_array = nn.get_weight()
            # collect weight array at the first ML process
            if rank == rank_ml[0]:
                weight_array_collect = np.empty((n_ml*weight_array.shape[0]), dtype=float)
            else:
                weight_array_collect = None
            comm_ml.Gather([weight_array, MPI.DOUBLE], [weight_array_collect, MPI.DOUBLE], root=0)
            t1 = time.time()
            # distribute the weight array to each PL process
            if rank == rank_ml[0]:
                if req_weight != None:
                    req_weight.Wait()
                weight_array_collect = np.concatenate((weight_array_collect, weight_array), axis=0)
                req_weight = comm_ml_pl.Iscatter([weight_array_collect, MPI.DOUBLE], [weight_array, MPI.DOUBLE], root=comm_ml_pl.Get_size()-1)
            # free memory
            del weight_array, weight_array_collect
            gc.collect()
            print("Weights sent to PL processes.")
            t2 = time.time()
            #tmp = None
            #ml_req = comm_ml_mg.Igather([ml_finished, MPI.LONG], tmp, root=RANK_MG)
            # process data received from MG
            new_coord = data_send[:retrain_step*n_atoms*3].reshape(retrain_step, n_atoms, 3)
            data_send = data_send[retrain_step*n_atoms*3:]
            new_energy = data_send[:retrain_step*n_states].reshape(retrain_step, n_states) * HtoEv
            new_force = data_send[retrain_step*n_states:].reshape(retrain_step, n_states, n_atoms, 3) * HtoEv * AToBohr
            # add new training data to training set
            nn.add_trainingset(new_coord, new_energy, new_force)
            
            #time_ml["gather_weight"].append(t1-t0)
            #time_ml["scatter_weight"].append(t2-t1)
            if save_progress:
                # save retraining progress (histroy + weights)
                nn.save_progress()
                #with open(os.path.join(ml_path, f'mltime_{rank}.json'), 'w') as fh:
                #    json.dump(time_ml, fh)
                save_progress = False
                print("model progress saved...")
                t_start = time.time()
        
        fstdout.close()
            
    
    # Manager Process (MG).
    # Communication with PL, MD, QM and ML processes
    if rank == RANK_MG:
        from component.ml_interface import masked_MAE, masked_r2
        # create directory to store data of MG processes
        #mg_path = os.path.join(directory, "process_manager")
        #if not os.path.exists(mg_path):
        #    os.makedirs(mg_path)
        # each rank writes to its own log file
        fstdout = open(os.path.join(mg_path, "mglog.txt"), 'w')
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start process manager process...")
        
        thresh = setting['manager']['std_thresh']    # threshold of std calculation for energy predictions
        assert rank == comm_ml_mg.Get_rank(), f"Error: rank {rank} is not rank 0 in comm_ml_mg"
        assert rank == comm_pl_mg.Get_rank(), f"Error: rank {rank} is not rank 0 in comm_pl_mg"
        assert rank == comm_md_mg.Get_rank(), f"Error: rank {rank} is not rank 0 in comm_md_mg"
        #ml_ready = True            # indicate if ML retraining is finished and is ready to update PL
        pl_updated = False         # indicate if PL models have finished updating weights
        req_ml_data = None         # used to check if retrain data has reached to ML process
        qm_free = list(rank_qm)    # list of ranks of QM processes that are idle
        qm_busy = {}               # dictionary of {QM ranks: DFT start time}
        
        qm_data_path = os.path.join(mg_path, 'data_to_qm.npy')
        ml_data_path = os.path.join(mg_path, 'data_to_ml.npy')
        test_res_path = os.path.join(mg_path, "testset_results")
        testset_eng_mae, testset_eng_r2, testset_std = [[] for i in range(0, model_total)], [[] for i in range(0, model_total)], []
        try:
            with open(qm_data_path, 'rb') as fh:
                coords_to_qm = list(np.load(fh))
                state_to_qm = list(np.load(fh))
        except:
            coords_to_qm = []      # buffer to store coordinates if no idle QM process
            state_to_qm = []       # buffer to store corresponding states of coordinates in coords_to_qm (dft calculation requires specifying the state)
            
        try:
            with open(ml_data_path, 'rb') as fh:
                coords_to_ml = list(np.load(fh))
                energy_to_ml = list(np.loat(fh))
                force_to_ml = list(np.load(fh))
        except:
            coords_to_ml = []      # list of coordinate to retrain ML models
            energy_to_ml = []      # list of energy to retrain ML models
            force_to_ml = []       # list of force to retrain ML models
        #data_to_ml = []            # list of coordinate/energy/force to retrain ML models
        
        
        with open(setting['passive']['test_path'], 'r') as fh:
            _, testset_eng, _ = json.load(fh)
        testset_eng = np.array(testset_eng, dtype=float) * HtoEv
        
        #try:
        #    with open(os.path.join(mg_path, "testset_results.json"), 'r') as fh:
        #        testset_eng_mae, testset_eng_r2, testset_std = json.load(fh)
        #except:
        #    testset_eng_mae, testset_eng_r2, testset_std = [[] for i in range(0, model_total)], [[] for i in range(0, model_total)], []
        
        #try:
        #    with open(os.path.join(mg_path, "mgtime.json"), "r") as fh:
        #        time_mg = json.load(fh)
        #except:
        #    time_mg = {
        #        "coord_from_md": [],
        #        "coord_to_pl": [],
        #        "pred_from_pl": [],
        #        "check_qm": [],
        #        "to_qm": [],
        #        "to_ml": [],
        #        "pred_to_md": []
        #        }
        
        t_start = time.time()
        save_progress = False
        save_progress_qm = False
        pl_comm_count = 0
        md_comm_count = 0
        while not stop_run:
            # save the progress after some time
            if time.time() - t_start >= update_time:
                save_progress = True
                save_progress_qm = True
                t_start = time.time()

            #update_pl = False
            #if not ml_ready:
            #    if req_ml_data == None:
            #        ml_finished = np.array([1,], dtype=int)
            #        tmp = np.empty((1+n_ml,), dtype=int)
            #        req_ml_ready = comm_ml_mg.Igather([ml_finished, MPI.LONG], tmp, root=RANK_MG)
            #        ml_status = MPI.Status()
            #    if req_ml_ready.Test(ml_status):
            #        print("ML retrain finished.")
            #        update_pl = True
            #        ml_ready = True
            #        req_ml_ready.Free()
            #        req_ml_ready = None
            #    else:
            #        update_pl = False
            
            # gather coordinates from MD process for PL prediction
            # data gathered from each MD process contains: current state (int) and coordinates (float, shape: (n_atoms, 3))
            t0 = time.time()
            coords_send = np.zeros((n_atoms*3+1,), dtype=float)
            coords_collect = np.empty(((n_md+1)*(n_atoms*3+1),), dtype=float)
            comm_md_mg.Gather([coords_send, MPI.DOUBLE], coords_collect, root=RANK_MG)
            md_comm_count += 1
            if md_comm_count >= 1.5 * data_count_save:
                md_save_data = 2.0
                md_comm_count = 0
            elif md_comm_count >= data_count_save:
                md_save_data = 1.0
            else:
                md_save_data = 0.0
            
            if pl_comm_count >= data_count_save:
                pl_save_data = 1
                pl_comm_count = 0
            else:
                pl_save_data = 0
            # reorganize received data to seperate current state and coordinates
            coords_collect = coords_collect.reshape(n_md+1, n_atoms*3+1)[1:]
            current_state = np.array(coords_collect[:,0], dtype=int)
            coords_collect = coords_collect[:,1:].reshape(n_md, n_atoms, 3)
            print(f"Data received from MD process: coordinates shape {coords_collect.shape}, current state shape {current_state.shape}")
            if pl_updated:
                #coords_to_pl = np.concatenate((np.ones((1,), dtype=float), coords_collect.flatten()), axis=0)
                coords_to_pl = [coords_collect, np.array(coords_to_qm), pl_save_data]
            else:
                #coords_to_pl = np.concatenate((np.zeros((1,), dtype=float), coords_collect.flatten()), axis=0)
                coords_to_pl = [coords_collect, None, pl_save_data]
            t1 = time.time()
            #time_mg['coord_from_md'].append(t1-t0)
                
            # broadcast coordinates to PL processes
            #comm_pl_mg.Bcast([coords_to_pl, MPI.DOUBLE], root=RANK_MG)
            coords_to_pl = comm_pl_mg.bcast(coords_to_pl, root=RANK_MG)
            t2 = time.time()
            #time_mg['coord_to_pl'].append(t2-t1)
            print("Coordinates and ML retraining status has been broadcasted to PL process.")
            energy_predictions = np.empty((model_total, n_md, n_states), dtype=float)
            force_predictions = np.empty((model_total, n_md, n_states, n_atoms, 3), dtype=float)
            if pl_updated:                
                data_count = testset_eng.shape[0]*testset_eng.shape[1] + len(coords_to_qm)*n_states + n_md*n_states*(1+n_atoms*3) + 1
                data_to_mg = np.zeros((data_count,), dtype=float)
                data_from_pl = np.empty((data_count*(model_total+1)), dtype=float)
                comm_pl_mg.Gather([data_to_mg, MPI.DOUBLE], [data_from_pl, MPI.DOUBLE], root=RANK_MG)
                data_from_pl = data_from_pl[data_count:]
                test_eng_pred = np.empty((model_total, testset_eng.shape[0], testset_eng.shape[1]), dtype=float)
                qm_eng_pred = np.empty((model_total, len(coords_to_qm), n_states), dtype=float)
                #energy_predictions = []
                #force_predictions = []
                for i in range(0, model_total):
                    #testset_eng_mae[i].append(float(data_from_pl[0]))
                    #testset_eng_r2[i].append(float(data_from_pl[1]))
                    #data_from_pl = data_from_pl[2:]
                    test_eng_pred[i] = data_from_pl[:testset_eng.shape[0]*testset_eng.shape[1]].reshape(testset_eng.shape[0], testset_eng.shape[1])
                    data_from_pl = data_from_pl[testset_eng.shape[0]*testset_eng.shape[1]:]
                    testset_eng_mae[i].append(float(masked_MAE(testset_eng, test_eng_pred[i])))
                    testset_eng_r2[i].append(float(masked_r2(testset_eng, test_eng_pred[i])))
                    qm_eng_pred[i] = data_from_pl[:len(coords_to_qm)*n_states].reshape(len(coords_to_qm), n_states)
                    data_from_pl = data_from_pl[len(coords_to_qm)*n_states:]
                    energy_predictions[i] = data_from_pl[:n_md*n_states].reshape(n_md, n_states)
                    data_from_pl = data_from_pl[n_md*n_states:]
                    force_predictions[i] = data_from_pl[:n_md*n_states*n_atoms*3].reshape(n_md, n_states, n_atoms, 3)
                    data_from_pl = data_from_pl[n_md*n_states*n_atoms*3:]
                    pl_updated = bool(int(data_from_pl[0]) == 1)
                    data_from_pl = data_from_pl[1:]
                testset_std.append(np.std(test_eng_pred, axis=0, ddof=1).tolist())
                if len(coords_to_qm) != 0:
                    qm_eng_std = np.std(qm_eng_pred, axis=0, ddof=1)
                    assert qm_eng_std.shape == (len(coords_to_qm), n_states)
                    qm_eng_idx_sorted = np.argsort(np.mean(qm_eng_std, axis=1), axis=0)
                    coords_to_qm = np.array(coords_to_qm, dtype=float)[qm_eng_idx_sorted]
                    state_to_qm = np.array(state_to_qm, dtype=int)[qm_eng_idx_sorted]
                    qm_eng_std = qm_eng_std[qm_eng_idx_sorted]
                    coords_to_qm = list(coords_to_qm[np.nonzero((qm_eng_std > thresh).any(axis=1))[0]])
                    state_to_qm = list(state_to_qm[np.nonzero((qm_eng_std > thresh).any(axis=1))[0]])
                if save_progress_qm:
                    # save the progress: performance on testset after update the PL
                    test_thread = threading.Thread(target=save_pickle, args=({test_res_path: [testset_eng_mae.copy(), testset_eng_r2.copy(), testset_std.copy()]},), daemon=True)
                    test_thread.start()
                    del testset_eng_mae, testset_eng_r2, testset_std
                    gc.collect()
                    testset_eng_mae, testset_eng_r2, testset_std = [[] for i in range(0, model_total)], [[] for i in range(0, model_total)], []
                    # save the progress: current coordinates and corresponding states waiting to be sent to QM
                    qm_thread = threading.Thread(target=save_np, args=({qm_data_path: [np.array(coords_to_qm, dtype=float), np.array(state_to_qm, dtype=int)]},), daemon=True)
                    qm_thread.start()
                    save_progress_qm = False
                    #with open(os.path.join(mg_path, "testset_results.json"), 'w') as fh:
                    #   json.dump([testset_eng_mae, testset_eng_r2, testset_std], fh)
                    # save the progress: current coordinates and corresponding states waiting to be sent to QM
                    #with open(qm_data_path, 'wb') as fh:
                    #    np.save(fh, np.array(coords_to_qm, dtype=float))
                    #    np.save(fh, np.array(state_to_qm, dtype=int))
                    #save_progress_qm = False
                
            else:
                # gather energy and force predictions from PL processes
                # data gathered from each PL process contains: energy predictions (n_md, n_states) and gradients (n_md, n_states, n_atoms, 3) and PL update state (1)
                data_count = n_md*n_states + n_md*n_states*n_atoms*3 + 1
                data_to_mg = np.zeros((data_count,), dtype=float)
                data_from_pl = np.empty(((model_total + 1) * data_count,), dtype=float)
                comm_pl_mg.Gather([data_to_mg, MPI.DOUBLE], [data_from_pl, MPI.DOUBLE], root=RANK_MG)
                # organize data received from PL process
                data_from_pl = data_from_pl[data_count:]
                #energy_predictions = []
                #force_predictions = []
                for i in range(0, model_total):
                    energy_predictions[i] = data_from_pl[:n_md*n_states].reshape(n_md, n_states)
                    force_predictions[i] = data_from_pl[n_md*n_states:n_md*n_states+n_md*n_states*n_atoms*3].reshape(n_md, n_states, n_atoms, 3)
                    data_from_pl = data_from_pl[n_md*n_states+n_md*n_states*n_atoms*3:]
                    pl_updated = bool(int(data_from_pl[0]) == 1)
                    data_from_pl = data_from_pl[1:]
            pl_comm_count += 1
            t3 = time.time()
            #time_mg['pred_from_pl'].append(t3-t2)
            #energy_predictions = np.array(energy_predictions).reshape(model_total, n_md, n_states)
            #force_predictions = np.array(force_predictions).reshape(model_total, n_md, n_states, n_atoms, 3)
            print(f"Energy received from PL: {energy_predictions.shape}")
            print(f"Force received from PL: {force_predictions.shape}")
            #data_count = model_per_node*n_md*n_states+1
            #energy_prediction_1 = eng_force_predictions[1:data_count]
            #data_count += model_per_node*n_md*n_states*n_atoms*3
            #force_prediction_1 = eng_force_predictions[1+len(energy_prediction_1):data_count]
            #data_count += model_per_node*n_md*n_states
            #energy_prediction_2 = eng_force_predictions[1+len(energy_prediction_1)+len(force_prediction_1):data_count]
            #force_prediction_2 = eng_force_predictions[1+len(energy_prediction_1)+len(force_prediction_1)+len(energy_prediction_2):]
            #energy_prediction_1 = energy_prediction_1.reshape(model_per_node, n_md, n_states)
            #force_prediction_1 = force_prediction_1.reshape(model_per_node, n_md, n_states, n_atoms, 3)
            #energy_prediction_2 = energy_prediction_1.reshape(model_per_node, n_md, n_states)
            #force_prediction_2 = force_prediction_1.reshape(model_per_node, n_md, n_states, n_atoms, 3)
            #energy_prediction = np.concatenate((energy_prediction_1, energy_prediction_2), axis=0)
            #del energy_prediction_1, energy_prediction_2
            #force_prediction = np.concatenate((force_prediction_1, force_prediction_2), axis=0)
            #del force_prediction_1, force_prediction_2
            
            # check std of energy predictions
            if model_total > 1:
                not_pass = eval_std(energy_predictions, thresh)    # index of predictions with high std
                print(f"Index of PL energy predictions with high standard deviation: {not_pass}")
            else:
                not_pass = np.array([], dtype=int)
                print("All predictions have passed the std check.")
            
            # send coordinates with high std predictions to QM processes for dft calculation
            # check working QM ranks and put ranks that are done with dft into idle list
            t4 = time.time()
            to_remove = []
            for i, t in qm_busy.items():
                if time.time() - t >= dft_wait_time:
                    dft_done = np.empty((1,), dtype=int)
                    tag_here = t_qm_mg[i - rank_qm[0]]
                    comm_world.Recv([dft_done, MPI.LONG], source=i, tag=tag_here)
                    if int(dft_done) == 1:
                        to_remove.append(i)
                        qm_free.append(i)
                        data_from_qm = np.empty((n_atoms*3 + n_states + n_states*n_atoms*3), dtype=float)
                        comm_world.Recv([data_from_qm, MPI.DOUBLE], source=i, tag=tag_here)
                        coords_to_ml.append(data_from_qm[:n_atoms*3].reshape(n_atoms,3))
                        energy_to_ml.append(data_from_qm[n_atoms*3:n_atoms*3+n_states].reshape(n_states,))
                        force_to_ml.append(data_from_qm[n_atoms*3+n_states:].reshape(n_states,n_atoms,3))
                        print("Data from QM rank {i} has arrived.")
                    else:
                        qm_busy[i] += 120
            for i in to_remove:
                qm_busy.pop(i)
            t5 = time.time()
            #time_mg['check_qm'].append(t5-t4)
            # save the progress: current coordinates, energy and force ready for ML
            if save_progress:
                ml_thread = threading.Thread(target=save_np, args=({ml_data_path: [np.array(coords_to_ml, dtype=float), np.array(energy_to_ml, dtype=float), np.array(force_to_ml, dtype=float)]},), daemon=True)
                ml_thread.start()
                dir_thread = threading.Thread(target=copy_dir, args=(directory, al_dir), daemon=True)
                dir_thread.start()
                
                #with open(ml_data_path, 'wb') as fh:
                #   np.save(fh, np.array(coords_to_ml, dtype=float))
                #    np.save(fh, np.array(energy_to_ml, dtype=float))
                #    np.save(fh, np.array(force_to_ml, dtype=float))
                save_progress = False
            
            #for i in qm_busy.keys():
            #    s = MPI.Status()
            #    if qm_busy[i].Test(s):
            #        coords_to_ml.append(data_to_ml[i][:n_atoms*3].reshape(n_atoms,3))
            #        energy_to_ml.append(data_to_ml[i][n_atoms*3:n_atoms*3+n_states].reshape(n_states,))
            #        force_to_ml.append(data_to_ml[i][n_atoms*3+n_states:].reshape(n_states,n_atoms,3))
            #        to_remove.append(i)
            #        qm_free.append(i)
            #        print("Data from QM rank {i} has arrived.")
            #for i in to_remove:
            #    data_to_ml.pop(i)
            #    qm_busy.pop(i)
            # add coordinates sent to QM process to the waiting list (coords_to_qm)
            if not_pass.shape[0] > 0:
                coords_to_qm += list(coords_collect[not_pass])
                state_to_qm += list(current_state[not_pass])
            # send each coordinates and corresponding state to each idle QM
            t6 = time.time()
            for i in range(0, min(len(coords_to_qm), len(qm_free))):
                data_to_qm = np.concatenate((np.array([state_to_qm.pop(0),]), coords_to_qm.pop(-1).flatten()), axis=0)
                qm_free_rank = qm_free.pop(0)
                tag_here = t_qm_mg[qm_free_rank - rank_qm[0]]
                comm_world.Send([data_to_qm, MPI.DOUBLE], dest=qm_free_rank, tag=tag_here)
                # data received from each QM process contains: coordinates (n_atoms, 3), energy (n_states), gradient (n_atoms, 3)
                #data_to_ml.append(np.empty((n_atoms*3 + n_states + n_states*n_atoms*3), dtype=float))
                #qm_busy[qm_free_rank] = comm_world.Irecv([data_to_ml[-1], MPI.DOUBLE], source=qm_free_rank, tag=tag_here)
                print(f"Coordinates and current state have been send to QM process {qm_free_rank}")
                qm_busy[qm_free_rank] = time.time()
            t7 = time.time()
            #time_mg['to_qm'].append(t7-t6)
            
            # check if ML has received new retraining data
            if req_ml_data != None:
                if req_ml_data.Test():
                    # retraining data have reached to ML
                    # update PL models
                    req_ml_data = None
                    
            # send coordinate/energy/force to ML processes for retraining
            if req_ml_data == None and len(coords_to_ml) >= retrain_step:
                #data_send = np.array([coords_to_ml[:retrain_step], energy_to_ml[:retrain_step], force_to_ml[:retrain_step]], dtype=float).flatten()
                data_send = np.concatenate((np.array(coords_to_ml[:retrain_step]).flatten(),np.array(energy_to_ml[:retrain_step]).flatten(),\
                                            np.array(force_to_ml[:retrain_step]).flatten()), axis=0)
                req_ml_data = comm_ml_mg.Ibcast([data_send, MPI.DOUBLE], root=RANK_MG)
                #ml_ready = False
                coords_to_ml = coords_to_ml[retrain_step:]
                energy_to_ml = energy_to_ml[retrain_step:]
                force_to_ml = force_to_ml[retrain_step:]
                print("Data have been sent to ML process for retraining.")
            t8 = time.time()
            #time_mg['to_ml'].append(t8-t7)
            
            # set energy/force predictions to be 0.0 where std are too high
            # so corresponding MD processes are notified upon receving predictions
            energy_predictions[:,not_pass] = np.zeros((n_states,), dtype=float)
            force_predictions[:,not_pass] = np.zeros((n_states, n_atoms, 3), dtype=float)
            assert energy_predictions.shape == (model_total, n_md, n_states),\
                "Errors: energy prediction shape not correct after std calculation."
            assert force_predictions.shape == (model_total, n_md, n_states, n_atoms, 3),\
                "Errors: force prediction shape not correct after std calculation."
            # calculate means of different model's predictions
            energy_predictions = np.mean(energy_predictions, axis=0)
            force_predictions = np.mean(force_predictions, axis=0)
            data_to_md = np.zeros((n_states+n_states*n_atoms*3+1,), dtype=float)
            for i in range(0, energy_predictions.shape[0]):
                #data_to_md.append(np.concatenate((energy_predictions[i].flatten(), force_predictions[i].flatten()), axis=0))
                data_to_md = np.concatenate((data_to_md, energy_predictions[i].flatten(), force_predictions[i].flatten()), axis=0)
                data_to_md = np.append(data_to_md, [md_save_data,], axis=0)
            # send energy/force back to MD processes
            data_recv = np.empty((n_states+n_states*n_atoms*3+1,), dtype=float)
            comm_md_mg.Scatter([data_to_md, MPI.DOUBLE], [data_recv, MPI.DOUBLE], root=RANK_MG)
            t9 = time.time()
            #time_mg['pred_to_md'].append(t9-t8)
            
            #if save_progress:
            #    with open(os.path.join(mg_path, "mgtime.json"), "w") as fh:
            #        json.dump(time_mg, fh)
            
        print("The End.")
        fstdout.close()
            
    
    
    

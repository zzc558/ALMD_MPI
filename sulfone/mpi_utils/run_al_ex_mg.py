#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:28:14 2023

@author: chen
"""

import numpy as np
from mpi4py import MPI
import sys, gc, os, time, shutil, threading

from component.al_setting import DEFAULT_AL_SETTING as setting


RANK_MG = 0                       # rank of manager process (MUST be fixed as rank 0)
RANK_EX = 1                       # rank of exchange process (MUST be fixed as rank 1)
HtoEv = 27.21138624598853
AToBohr = 1.889725989

def eval_std(data_array, threshold_up, threshold_low):
    std = np.std(data_array, axis=0, ddof=1)
    return np.where((std > threshold_up).any(axis=1))[0], std[np.where((std > threshold_low).any(axis=1))[0]]

def send_to_mg(greq, coords_to_mg, states_to_mg, req_ex_mg):
    coords_to_mg = np.array(coords_to_mg, dtype=float)
    states_to_mg = np.array(states_to_mg, dtype=float)
    assert coords_to_mg.shape[0] == states_to_mg.shape[0], "number of coordinates doesn't match number of states in the buffer to MG."
    data_to_mg = np.concatenate((coords_to_mg.flatten(), states_to_mg.flatten()), axis=0)
    req_ex_mg = comm_world.Isend([data_to_mg, MPI.DOUBLE], dest=RANK_MG, tag=t_ex_mg)
    return greq.Complete()


if __name__ == "__main__":
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
    run_al = global_setting['run_al']      # whether to run active learning or just MD propagation
    test_path = setting['passive']['test_path']
    
    rank_pl = tuple(range(2, n_pl+2))      # list of ranks of passive learner processes
    rank_md = tuple(range(n_pl+2, n_md+n_pl+2))    # list of ranks of molecular dynamic processes
    rank_qm = tuple(range(n_md+n_pl+2, n_qm+n_md+n_pl+2))    # list of ranks of quantum mechanic processes
    rank_ml = tuple(range(n_qm+n_md+n_pl+2, n_ml+n_qm+n_md+n_pl+2))    # list of ranks of machine learner processes
    
    # molecule information
    elements = global_setting['elements']
    n_atoms = len(elements)
    n_states = global_setting['n_states']
    # directory to store all the scratches/results
    al_dir = global_setting['res_dir']
    #directory = os.path.abspath(os.path.join(tmp_dir, al_dir))
    directory = os.path.abspath(al_dir)
    #res_dir = global_setting['res_dir']
    #res_dir = 'results/TestRun_dev'
    mg_path = os.path.join(directory, "process_manager")
    ex_path = os.path.join(directory, "exchange")
    md_path = os.path.join(directory, "molecular_dynamic")
    md_data_path = os.path.join(md_path, "trajData")
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
        os.makedirs(ex_path, exist_ok=True)
        os.makedirs(md_path, exist_ok=True)
        os.makedirs(md_data_path, exist_ok=True)
        os.makedirs(ml_path, exist_ok=True)
        os.makedirs(qm_path, exist_ok=True)
        os.makedirs(pl_path, exist_ok=True)
    t_pl_ex = 0                            # mpi tag for communication between PL and EX process
    t_md_ex = 1                            # mpi tag for communication between MD and EX process
    t_ex_mg = 2                            # mpi tag for communication between EX and MG process
    t_ml_mg = 3                            # mpi tag for communication between ML and MG process
    t_ml_pl = 4                            # mpi tag for communication between ML and PL process
    t_ml = 5                               # mpi tag for communication among ML processes
    t_md = 6
    t_pl = 7
    t_qm_mg = list(range(8, n_qm+8))       # mpi tag for communication between QMs and MG process
    group_world = comm_world.Get_group()
    # create communicator to pass message between PL processes and manager process
    group_pl_ex = group_world.Incl([RANK_EX,] + list(rank_pl))
    comm_pl_ex = comm_world.Create_group(group_pl_ex, tag=t_pl_ex)
    # create communicator to pass message between MD processes and manager process
    group_md_ex = group_world.Incl([RANK_EX,] + list(rank_md))
    comm_md_ex = comm_world.Create_group(group_md_ex, tag=t_md_ex)
    # create communicator to pass message between ML processes and manager process
    group_ml_mg = group_world.Incl([RANK_MG,] + list(rank_ml))
    comm_ml_mg = comm_world.Create_group(group_ml_mg, tag=t_ml_mg)
    # create communicator to pass weights between ML process and PL process
    group_ml_pl = group_world.Incl([rank_ml[0],] + list(rank_pl))
    comm_ml_pl = comm_world.Create_group(group_ml_pl, tag=t_ml_pl)
    # create communicator to collect weights among ML processes
    group_ml = group_world.Incl(list(rank_ml))
    comm_ml = comm_world.Create_group(group_ml, tag=t_ml)
    # Create communicator among MD processes for writing trajectory data to the disk
    group_md = group_world.Incl(list(rank_md))
    comm_md = comm_world.Create_group(group_md, tag=t_md)
    # Create communicator among PL processes for writing time data to the disk
    group_pl = group_world.Incl(list(rank_pl))
    comm_pl = comm_world.Create_group(group_pl, tag=t_pl)
    
    assert size == len(rank_pl)+len(rank_md)+len(rank_qm)+len(rank_ml)+2,\
        "Error: number of processes not equal to size of ranks"
    stop_run = False
    
    # Passive Learner Process (PL)
    # Recive coordinates from MD through MG, make predictions and send back to MD through MG
    # Copy new model and scaler weights from ML process and update models
    if rank in rank_pl:
        # each rank writes to its own log file
        while not os.path.exists(pl_path):
            time.sleep(1)
        pl_log = os.path.join(pl_path, f"pllog_{rank}.txt")
        mode_log = 'a' if os.path.exists(pl_log) else 'w'
        fstdout = open(pl_log, mode_log)
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start passive learner process...")
        
        # import library
        from component.ml_interface import MLForAl
        from component.mpi_utils import query_fn, free_fn, cancel_fn, save_time_data
        
        # read settings and initilize models
        pl_setting = setting['passive']
        gpu_list = pl_setting['gpu_list']
        model_name = pl_setting['model_name']
        model_path = pl_setting['path']
        model_hyper = setting['model_hyper']
        ml_source = setting['ml']['path']
        model_index = (comm_pl.Get_rank() - 1) % gpu_per_node
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
        nn = MLForAl(method='nn', kwargs=pl_input)
        
        req_weight = None    # request used to receive model weights from ML process
        save_thread = None   # threading object for writing PL time to disk
        
        # record the running time of each task
        ftime_pl = os.path.join(pl_path, 'pltime.mpi')
        time_keys = ['bcast', 'predict', 'gather', 'update', 'save']
        time_pl = {
            'bcast': [],
            'predict': [],
            'gather': [],
            'update': [],
            'save': [],
            }

        while not stop_run:
            # receive new weights from ML process
            t0 = time.time()
            if req_weight is None:
                weight_array_collect = None
                weight_array = np.empty((nn.get_num_weight(),), dtype=float)
                req_weight = comm_ml_pl.Iscatter([weight_array_collect, MPI.DOUBLE], [weight_array, MPI.DOUBLE], root=0)
                print("Wait for new weights from ML...")
            elif req_weight.Test():
                print("New weights arrived...")
                req_weight = None
                nn.update(weight_array)
                del weight_array
                gc.collect()
                print("Model updated.")
            t1 = time.time()
            
            # receive coordinates from EX process
            # shape of data from EX: coordinates (number of MD, number of atoms, 3) + save_progess flag (1)
            data_to_pl = np.empty((n_md*n_atoms*3+1), dtype=float)
            comm_pl_ex.Bcast([data_to_pl, MPI.DOUBLE], root=0)
            coords = data_to_pl[:-1].reshape(n_md, n_atoms, 3)
            save_progress = data_to_pl[-1]
            print("Received coordinates from EX.")
            t2 = time.time()
            
            # predict energies and forces
            eng_pred, force_pred = nn.predict_pl(coords)
            eng_pred = eng_pred[:,:n_states]
            force_pred = force_pred[:,:n_states]
            t3 = time.time()
            
            # send predictions to EX process
            data_to_ex = np.append(eng_pred.flatten(), force_pred.flatten(), axis=0)
            data_collected = None
            comm_pl_ex.Gather([data_to_ex, MPI.DOUBLE], [data_collected, MPI.DOUBLE], root=0)
            print("Predictions have been sent to EX.")
            t4 = time.time()
            
            # free memory
            del eng_pred, force_pred
            gc.collect()
            
            # save time records
            if save_progress == 1.0 and (save_thread is None or not save_thread.is_alive()):
                save_progress = 0.0
                
                # save time data using threading
                if os.path.exists(ftime_pl):
                    amode = MPI.MODE_APPEND|MPI.MODE_WRONLY
                else:
                    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
                pl_write_request = MPI.Grequest.Start(query_fn, free_fn, cancel_fn)
                save_thread = threading.Thread(target=save_time_data, name=f"save_time_{rank}",\
                                               args=(pl_write_request, time_pl, time_keys, comm_pl, ftime_pl, amode), daemon=True)
                save_thread.start()
                
                #data_save = np.array([], dtype=float)
                #for k in ['bcast', 'predict', 'gather', 'update', 'save']:
                #    data_save = np.append(data_save, time_pl[k], axis=0)
                #    data_save = np.append(data_save, [-1,], axis=0)
                #    time_pl[k] = []
                
                #if os.path.exists(ftime_pl):
                #    amode = MPI.MODE_APPEND|MPI.MODE_WRONLY
                #else:
                #    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
                #fh = MPI.File.Open(comm_pl, ftime_pl, amode)
                #displacement = fh.Get_size()    # number of bytes to be skipped from the start of the file
                #etype=MPI.DOUBLE    # basic unit of data access
                #filetype=None     # specifies which portion of the file is visible to the process
                # MPI noncontiguous and collective writing
                #fh.Set_view(displacement, etype, filetype)
                #fh.Write_ordered([data_save, MPI.DOUBLE])
                #fh.Close()
                print("Time data saved.")
            t5 = time.time()
            
            time_pl['update'].append(t1 - t0)
            time_pl['bcast'].append(t2 - t1)
            time_pl['predict'].append(t3 - t2)
            time_pl['gather'].append(t4 - t3)
            time_pl['save'].append(t5 - t4)
                
        print("The End.")
        fstdout.close()
            
    # Passive Molecular Dynamic (MD)
    # Propagate trajectories. Send coordinates to PL through MG
    if rank in rank_md:
        # each rank writes to its own log file
        while not os.path.exists(md_path):
            time.sleep(1)
        md_log = os.path.join(md_path, f"mdlog_{rank}.txt")
        mode_log = 'a' if os.path.exists(md_log) else 'w'
        fstdout = open(md_log, mode_log)
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start molecular dynamic process...")
        
        from component.md_utils import read_inicond, load_input, set_aimd
        from component.mpi_utils import save_np
        from PyRAI2MD.variables import ReadInput
        
        md_setting = setting['md']
        initial_cond_path = md_setting['initial_cond_path']
        md_input = os.path.abspath(md_setting['input_file'])
        
        job_keywords = ReadInput(load_input(md_input))    # input for MD trajectory
        
        # load initial conditions
        with open(initial_cond_path, 'rb') as fh:
            init_coord = np.load(fh)
            init_velc = np.load(fh)
            root = np.load(fh)
        root += 1
        init_idx = np.array(range(0, init_coord.shape[0]), dtype=int)
            
        # record trajectory data
        step_keys = ['coord', 'energy', 'force', 'state']    # list of keys of traj_data to data related to each time step
        traj_keys = ['termin',]    # list of keys of traj_data to data related to trajectories
        traj_data = {
            'energy': [],
            'force': [],
            'coord': [],
            'state': [],
            'termin': [],
            }
            
        os.chdir(md_path)
        save_progress = 0.0
        save_iter = 0
        save_thread = None   # threading object for writing trajectory data to disk
        while not stop_run:
            traj_dir = os.path.join(md_path, f"traj_{rank}")
            os.makedirs(traj_dir, exist_ok=True)
            os.chdir(traj_dir)
            # record trajectory data
            for k in traj_data.keys():
                if k != 'termin':
                    traj_data[k].append([])
            
            # randomly sample initial conditions
            i = np.random.choice(init_idx, size=1)
            while root[i] >= n_states+1:
                i = np.random.choice(init_idx, size=1)
            
            # set up MD trajectory
            mol = read_inicond(init_coord[i], init_velc[i], elements)[0]
            aimd = set_aimd(mol, job_keywords, md_setting)
            
            # run one trajectory
            traj_status = 0
            while aimd.traj.iter < aimd.step_left:
                aimd.traj.iter += 1
                
                # send state and coordinate to EX for energy and force prediction
                current_state, coord = aimd.propagate_step_one()
                data_to_ex = np.append([float(current_state),], coord.flatten(), axis=0)
                data_collected = None
                comm_md_ex.Gather([data_to_ex, MPI.DOUBLE], [data_collected, MPI.DOUBLE], root=0)
                
                # save progress when waiting for predictions
                if save_progress == 1.0 and (save_thread is None or not save_thread.is_alive()):
                    save_progress = 0.0
                    
                    # save trajectory data as numpy array with threading
                    ftraj_md = os.path.join(md_data_path, f"trajdata_{rank}.mpi")
                    mode = 'ab' if os.path.exists(ftraj_md) else 'wb'
                    save_thread = threading.Thread(target=save_np, name=f"save_traj_{rank}",\
                                                   args=(traj_data, step_keys, traj_keys, ftraj_md, mode, aimd.traj.velo, aimd.traj.iter), daemon=True)
                    save_thread.start()
                    
                    #data_save = generate_save_data(traj_data, ['coord', 'energy', 'force', 'state'], ['termin',])
                    
                    ## DEBUG: test saving data
                    #ftraj_md = os.path.join(md_data_path, f"trajdata_{rank}.mpi")
                    #mode = 'ab' if os.path.exists(ftraj_md) else 'wb'
                    #with open(ftraj_md, mode) as fh:
                    #    np.save(fh, data_save)
                    #del data_save
                    #gc.collect()

                    #data_save = np.array([], dtype=float)
                    #for k in ['coord', 'energy', 'force', 'state']:
                    #    data_save = np.concatenate((data_save, np.array(traj_data[k][:-1], dtype=float).flatten(), [-1.0,]), axis=0)
                    #    del traj_data[k][:-1]
                    #data_save = np.concatenate((data_save, np.array(traj_data['termin'], dtype=float), [-1.0,]), axis=0)
                    #traj_data['termin'] = []
                    #gc.collect()
                    
                    #ftraj_md = os.path.join(md_data_path, f"trajdata{save_iter}.mpi")
                    #if os.path.exists(ftraj_md):
                    #    amode = MPI.MODE_APPEND|MPI.MODE_WRONLY
                    #else:
                    #    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
                    #fh = MPI.File.Open(comm_md, ftraj_md, amode)
                    #displacement = fh.Get_size()    # number of bytes to be skipped from the start of the file
                    #etype=MPI.DOUBLE    # basic unit of data access
                    #filetype=None     # specifies which portion of the file is visible to the process
                    # MPI noncontiguous and collective writing
                    #fh.Set_view(displacement, etype, filetype)
                    #fh.Write_ordered([data_save, MPI.DOUBLE])
                    #fh.Close()
                    #del data_save
                    #gc.collect()
                    #save_iter += 1
                
                # receive energy and force prediction from EX
                data_to_md = None
                data_recv = np.empty((n_states+n_states*n_atoms*3+1,), dtype=float)
                comm_md_ex.Scatter([data_to_md, MPI.DOUBLE], [data_recv, MPI.DOUBLE], root=0)
                eng_pred = data_recv[:n_states].reshape(n_states,) / HtoEv
                force_pred = data_recv[n_states:-1].reshape(n_states, n_atoms, 3) / (HtoEv * AToBohr)
                save_progress = data_recv[-1]
                
                # check eng_pred and force_pred
                if (eng_pred == 0).all() and (force_pred == 0).all():
                    traj_status = 4    # indication of termination because of high STD
                    break
                
                # propagate trajectory with energy and force predictions
                traj_status = aimd.propagate_step_two(eng_pred, force_pred)
                
                # keep record of trajectory data
                traj_data['coord'][-1].append(np.copy(coord))
                traj_data['energy'][-1].append(np.copy(eng_pred))
                traj_data['force'][-1].append(np.copy(force_pred))
                traj_data['state'][-1].append(current_state)
                
                # free memory
                del data_recv, coord, eng_pred, force_pred, current_state
                gc.collect()
                
                if traj_status != 0:
                    # trajectory terminated abnormally
                    break
            
            # record trajectory termination status
            traj_data['termin'].append(traj_status)
            os.chdir(md_path)
            shutil.rmtree(traj_dir)
        
        print("The End.")
        fstdout.close()
            
    # Quantum Mechanics process (QM)
    # Receive geometries from MG and run DFT calculations
    if rank in rank_qm:
        # each rank writes to its own log file
        while not os.path.exists(qm_path):
            time.sleep(1)
        qm_log = os.path.join(qm_path, f"qmlog_{rank}.txt")
        mode_log = 'a' if os.path.exists(qm_log) else 'w'
        fstdout = open(qm_log, mode_log)
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start molecular dynamic process...")
        
        from component.qm_interface import QM_MPI
        
        tag_here = t_qm_mg[rank-rank_qm[0]]    # MPI tag for this QM process
        qm = QM_MPI(tmp_dir, rank, elements, n_states)
        
        while not stop_run:
            # receive coordinates from MG process
            coord_from_mg = np.empty((1+n_atoms*3,), dtype=float)
            comm_world.Recv([coord_from_mg, MPI.DOUBLE], source=RANK_MG, tag=tag_here)
            current_state = int(coord_from_mg[-1])
            coord_from_mg = coord_from_mg[:-1].reshape(n_atoms, 3)
            print("Received coordinates from MG process for DFT calculation.")
            # run dft calculations
            results = {}
            greq = qm.mpi_dft(coords=coord_from_mg, grad=True, num_ex=n_states-1, current_state=current_state, results=results)
            
            # check if the DFT calculation has finished
            while not greq.Test():
                time.sleep(120)
                
            # organize the dft calculation results
            dft_energy = np.array(results['energy'], dtype=float)
            assert current_state == results['current_n'], "Error: check the current state sent to dft calculation and received."
            dft_force = np.zeros((n_states, n_atoms, 3), dtype=float)
            if not results['gradient'] is None:
                dft_force[current_state] = np.array(results['gradient'], dtype=float)
            print("Done with the DFT calculation.")
            
            # send energy and force back to MG process
            comm_world.Send([np.concatenate((coord_from_mg.flatten(), dft_energy.flatten(), dft_force.flatten()),axis=0), MPI.DOUBLE], dest=RANK_MG, tag=tag_here)
            print("Data have been send back to MG process.")
        
        print("The End.")
        fstdout.close()
        
    # Machine learning Process (ML)
    # Receive coordinates, energy and force from MG
    # Retrain the model
    # Notify MG once finishing retraining
    if rank in rank_ml:
        from component.ml_interface import MLForAl
        # each rank writes to its own log file
        while not os.path.exists(ml_path):
            time.sleep(1)
        ml_log = os.path.join(ml_path, f"mllog_{rank}.txt")
        mode_log = 'a' if os.path.exists(ml_log) else 'w'
        fstdout = open(ml_log, mode_log)
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

        ml_input = {
            'gpu_index': model_index,
            'model_index': model_index,
            'model_dir': model_path,
            'model_name': 'eg',
            'hyper': model_hyper,
            'source': None,
            'mode': 'retrain'
            }
        
        # keep the test set to test the model after retraining
        #with open(test_path, 'r') as fh:
        #    c, e, g = json.load(fh)
        #c = np.array(c)
        #c = np.array(c[:,:,1:], dtype=float)
        #e = np.array(e, dtype=float) * HtoEv
        #g_shape = (e.shape[0], e.shape[1], c.shape[1], c.shape[2])
        #g_all = np.zeros(g_shape, dtype=float)
        #for i in range(0, len(g)):
        #    for s in g[i]:
        #        # s[0] indicate the state of gradient: 0 -> S0, 1 -> S1, ...
        #        # s[1] contains the value of gradient
        #        g_all[i][int(s[0])] = np.array(s[1], dtype=float) * HtoEv * AToBohr
        #ml_testset = [c, e, g_all]
        
        print(f"Rank {rank}: Initilize the model...")
        nn = MLForAl(method='nn', kwargs=ml_input)
        
        # wait for the first set of retraining data before start retraining
        data_from_mg = np.empty((retrain_step*(n_atoms*3+n_states+n_states*n_atoms*3)+1,), dtype=float)
        req_ml_data = comm_ml_mg.Ibcast([data_from_mg, MPI.DOUBLE], root=0)
        req_ml_data.Wait()
        # organize data received from MG
        new_coord = data_from_mg[:retrain_step*n_atoms*3].reshape(retrain_step, n_atoms, 3)
        data_from_mg = data_from_mg[retrain_step*n_atoms*3:]
        new_energy = data_from_mg[:retrain_step*n_states].reshape(retrain_step, n_states) * HtoEv
        new_force = data_from_mg[retrain_step*n_states:-1].reshape(retrain_step, n_states, n_atoms, 3) * HtoEv * AToBohr
        qm_buffer_size = data_from_mg[-1]
        # add new training data to training set
        nn.add_trainingset(new_coord, new_energy, new_force)
        del new_coord, new_energy, new_force
        if qm_buffer_size > 1:
            coords_to_qm = np.empty((qm_buffer_size*n_atoms*3,), dtype=float)
            comm_ml_mg.Bcast([coords_to_qm, MPI.DOUBLE], root=0)
            coords_to_qm = coords_to_qm.reshape(qm_buffer_size, n_atoms, 3)
            eng_pred, _ = nn.predict_ml(coords_to_qm)
            data_from_ml = None
            comm_ml_mg.Gather([eng_pred[:,:n_states].flatten(), MPI.DOUBLE], [data_from_ml, MPI.DOUBLE], root=0)
            del eng_pred, _
        gc.collect()

        while not stop_run:
            # receive coordinates, energy and force from MG process
            data_from_mg = np.empty((retrain_step*(n_atoms*3+n_states+n_states*n_atoms*3)+1,), dtype=float)
            req_ml_data = comm_ml_mg.Ibcast([data_from_mg, MPI.DOUBLE], root=0)
            
            # starting retraining with current training set
            # stop retraining when new training data arrive
            nn.retrain(req_ml_data)
            
            # when retraining finished or early stopping
            # wait until new retraining data arrive
            req_ml_data.Wait()
            print("Retraining finished.")
            
            # receive coordinates in the QM buffer of MG, predic energy and return to MG
            qm_buffer_size = data_from_mg[-1]
            if qm_buffer_size > 1:
                coords_to_qm = np.empty((qm_buffer_size*n_atoms*3,), dtype=float)
                comm_ml_mg.Bcast([coords_to_qm, MPI.DOUBLE], root=0)
                coords_to_qm = coords_to_qm.reshape(qm_buffer_size, n_atoms, 3)
                eng_pred, _ = nn.predict_ml(coords_to_qm)
                data_from_ml = None
                comm_ml_mg.Gather([eng_pred[:,:n_states].flatten(), MPI.DOUBLE], [data_from_ml, MPI.DOUBLE], root=0)
                # free memory
                del eng_pred, _
                gc.collect()
            
            # get weight array
            weight_array = nn.get_weight()
            
            # collect weight array at the ML process with the lowest rank
            if rank == rank_ml[0]:
                weight_array_collect = np.empty((n_ml*weight_array.shape[0]), dtype=float)
            else:
                weight_array_collect = None
            comm_ml.Gather([weight_array, MPI.DOUBLE], [weight_array_collect, MPI.DOUBLE], root=0)
            
            # distribute the weight array to each PL process
            if rank == rank_ml[0]:
                if req_weight != None:
                    req_weight.Wait()
                weight_array_collect = np.concatenate((weight_array, weight_array_collect), axis=0)
                req_weight = comm_ml_pl.Iscatter([weight_array_collect, MPI.DOUBLE], [weight_array, MPI.DOUBLE], root=0)
            # free memory
            del weight_array, weight_array_collect
            gc.collect()
            print("Weights sent to PL processes.")
            
            # organize received data
            new_coord = data_from_mg[:retrain_step*n_atoms*3].reshape(retrain_step, n_atoms, 3)
            data_from_mg = data_from_mg[retrain_step*n_atoms*3:]
            new_energy = data_from_mg[:retrain_step*n_states].reshape(retrain_step, n_states) * HtoEv
            new_force = data_from_mg[retrain_step*n_states:-1].reshape(retrain_step, n_states, n_atoms, 3) * HtoEv * AToBohr
            # add new training data to training set
            nn.add_trainingset(new_coord, new_energy, new_force)
            # free memory
            del new_coord, new_energy, new_force, data_from_mg
            gc.collect()
            
            # save the weight, dataset and history
            save_progress = False
            print("model progress saved...")
            
        fstdout.close()
            
    # Exchange Process (EX).
    # Communication with PL, MD and MG processes
    if rank == RANK_EX:
        while not os.path.exists(ex_path):
            time.sleep(1)
        # each rank writes to its own log file
        ex_log = os.path.join(ex_path, "exlog.txt")
        mode_log = 'a' if os.path.exists(ex_log) else 'w'
        fstdout = open(ex_log, mode_log)
        sys.stderr = fstdout
        sys.stdout = fstdout
        
        # import library
        from component.mpi_utils import save_np, query_fn, free_fn, cancel_fn
        
        thresh_up = setting['manager']['std_thresh']    # threshold of std calculation for energy predictions
        thresh_low = setting['manager']['std_thresh_record']    # threshold of recording std calculation results
        
        high_std_record_path = os.path.join(ex_path, "high_std_record.npy")    # file to record STD values above thresh_low
        high_std_record = []    # record STD values above thresh_low
        
        buffer_path  = os.path.join(ex_path, "to_mg_buffer.npy")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as fh:
                coords_to_mg = np.load(fh).tolist()
                states_to_mg = np.load(fh).tolist()
        else:
            coords_to_mg = []    # coordinates to be sent to MG for QM calculation
            states_to_mg = []    # states to be sent to MG for QM calculation
        
        req_ex_mg = None    # MPI.Request object for communication between EX and MG processes
        save_std_thread = None    # threading object for writing std records
        save_mg_thread = None    # threading object for writing data to MG process
        
        stop_run = False    # flag that indicates termination of active learning (not implemented yet)
        time_start = time.time()    # record the starting point time for progress saving
        print("Active learning starts...")
        while not stop_run:
            if time.time() - time_start >= update_time:
                # set the save_progress flag to true and reset the starting point time
                save_progress = 1.0
                time_start = time.time()
            else:
                save_progress = 0.0
                
            # collect coordinates and state from all MD processes for PL prediction
            # shape of data from each MD process: current state: 1 + coordinates: (number of atoms, 3)
            data_to_ex = np.zeros((n_atoms*3+1,), dtype=float)
            data_collected = np.empty(((n_md+1)*(n_atoms*3+1)), dtype=float)
            comm_md_ex.Gather([data_to_ex, MPI.DOUBLE], [data_collected, MPI.DOUBLE], root=0)
            print("Coordinates have been received from MD processes.")
            
            # organize data collected from all MD processes
            data_collected = data_collected.reshape(n_md+1, n_atoms*3+1)[1:]    # remove the data_to_ex from EX process itself
            current_state = np.array(data_collected[:,0], dtype=int)
            coords_collect = data_collected[:,1:].reshape(n_md, n_atoms, 3)
            
            # free memory
            del data_collected
            gc.collect()
            
            # broadcast coordinates and save_progress flag to PL processes for prediction
            data_to_pl = np.append(coords_collect.flatten(), [save_progress,], axis=0)
            comm_pl_ex.Bcast([data_to_pl, MPI.DOUBLE], root=0)
            print("Coordinates have been sent to PL processes.")
            
            # save the meta data during PL making predictions
            if save_progress == 1.0:
                # save meta data with threading
                # save STD value record
                if len(high_std_record) > 0 and (save_std_thread is None or not save_std_thread.is_alive()):
                    save_progress = 0.0
                    mode = 'ab' if os.path.exists(high_std_record_path) else 'wb'
                    save_std_thread = threading.Thread(target=save_np, name=f"save_std_{rank}",\
                                                   args=([np.array(high_std_record, dtype=float),], [], [], high_std_record_path, mode), daemon=True)
                    save_std_thread.start()
                    # free memory
                    del high_std_record
                    gc.collect()
                    high_std_record = []
                    print("STD records saved.")
                # save coords_to_mg and states_to_mg
                if len(coords_to_mg) > 0 and (save_mg_thread is None or not save_mg_thread.is_alive()):
                    save_progress = 0.0
                    save_mg_thread = threading.Thread(target=save_np, name=f"save_mg_{rank}",\
                                                   args=([np.array(coords_to_mg, dtype=float), np.array(states_to_mg, dtype=float)], [], [], buffer_path, 'wb'), daemon=True)
                    save_mg_thread.start()
                    print("Buffer saved.")
                # save STD value record
                #if len(high_std_record) > 0:
                #    mode = 'ab' if os.path.exists(high_std_record_path) else 'wb'
                #    with open(high_std_record_path, mode) as fh:
                #        np.save(fh, np.array(high_std_record, dtype=float))
                #    del high_std_record
                #    gc.collect()
                #    high_std_record = []
                #    print("STD records saved.")
                # save coords_to_mg and states_to_mg
                #if len(coords_to_mg) > 0:
                #    with open(buffer_path, 'wb') as fh:
                #        np.save(fh, np.array(coords_to_mg, dtype=float))
                #        np.save(fh, np.array(states_to_mg, dtype=float))
                #    print("Buffer saved.")
            
            # gather energy and force predictions from all PL processes
            # shape of data from each PL process: energy predictions: (number of MDs, number of states) + force predictions: (number of MDs, number of states, number of atoms, 3)
            data_count = n_md * (n_states + n_states * n_atoms * 3)
            data_to_ex = np.zeros((data_count,), dtype=float)
            data_collected = np.empty(((n_pl + 1) * data_count,), dtype=float)
            comm_pl_ex.Gather([data_to_ex, MPI.DOUBLE], [data_collected, MPI.DOUBLE], root=0)
            print("Predictions have been received from PL processes")
            
            # organize data collected from all PL processes
            eng_predictions = np.empty((model_total, n_md, n_states), dtype=float)
            force_predictions = np.empty((model_total, n_md, n_states, n_atoms, 3), dtype=float)
            data_collected = data_collected[data_count:]
            for i in range(0, model_total):
                eng_predictions[i] = data_collected[:data_count][:n_md*n_states].reshape(n_md, n_states)
                force_predictions[i] = data_collected[:data_count][n_md*n_states:].reshape(n_md, n_states, n_atoms, 3)
                data_collected = data_collected[data_count:]
                
            # evaluate STD of energy predictions
            not_pass, std_to_record = eval_std(eng_predictions, thresh_up, thresh_low)
            high_std_record += std_to_record.tolist()
            
            # put into buffer coordinates and states with high energy prediction STD that will be sent to MG process
            if run_al and not_pass.shape[0] > 0:
                coords_to_mg += coords_collect[not_pass].tolist()
                states_to_mg += current_state[not_pass].tolist()
                
            # set high STD predictions to 0 to notify corresponding MD process
            eng_predictions[:,not_pass] = np.zeros((n_states,), dtype=float)    
            force_predictions[:,not_pass] = np.zeros((n_states, n_atoms, 3), dtype=float)
            # average energy and force predictions from all PL
            eng_predictions = np.mean(eng_predictions, axis=0)
            force_predictions = np.mean(force_predictions, axis=0)
            # organize the data for distributing to MD processes
            # data count of data to each MD process: energy (number of states) + force (number of states, number of atoms, 3) + save_progress flag (1)
            data_to_md = np.zeros((n_states+n_states*n_atoms*3+1,), dtype=float)
            for i in range(0, n_md):
                data_to_md = np.concatenate((data_to_md, eng_predictions[i].flatten(), force_predictions[i].flatten(), [save_progress,]), axis=0)
            # distribute data to MD processes
            data_recv = np.empty((n_states+n_states*n_atoms*3+1,), dtype=float)
            comm_md_ex.Scatter([data_to_md, MPI.DOUBLE], [data_recv, MPI.DOUBLE], root=0)
            print("Predictions have been distributed to MD processes.")
            
            # free memory
            del eng_predictions, force_predictions, data_to_md
            gc.collect()
            
            # non-blocking send save_progress flag, coordinates and states in buffer to MG process for QM calculation
            if run_al and len(coords_to_mg) > 0 and (req_ex_mg is None or req_ex_mg.Test()):
                coords_to_mg = np.array(coords_to_mg, dtype=float)
                states_to_mg = np.array(states_to_mg, dtype=float)
                assert coords_to_mg.shape[0] == states_to_mg.shape[0], "number of coordinates doesn't match number of states in the buffer to MG."
                data_to_mg = np.concatenate((coords_to_mg.flatten(), states_to_mg.flatten()), axis=0)
                req_ex_mg = comm_world.Isend([data_to_mg, MPI.DOUBLE], dest=RANK_MG, tag=t_ex_mg)
                
                # free memory
                del coords_to_mg, states_to_mg
                gc.collect()
                coords_to_mg, states_to_mg = [], []
                
        print("The End.")
        fstdout.close()
    
    # Manager Process (MG).
    # Communication with PL, MD, QM and ML processes
    if rank == RANK_MG:
        # each rank writes to its own log file
        mg_log = os.path.join(mg_path, "mglog.txt")
        mode_log = 'a' if os.path.exists(mg_log) else 'w'
        fstdout = open(mg_log, mode_log)
        sys.stderr = fstdout
        sys.stdout = fstdout

        print("Start process manager process...")
        
        thresh = setting['manager']['std_thresh']    # threshold of std calculation for energy predictions
        assert rank == comm_ml_mg.Get_rank(), f"Error: rank {rank} is not rank 0 in comm_ml_mg"

        #ml_ready = True            # indicate if ML retraining is finished and is ready to update PL
        pl_updated = False         # indicate if PL models have finished updating weights
        req_ml_data = None         # used to check if retrain data has reached to ML process
        qm_free = list(rank_qm)    # list of ranks of QM processes that are idle
        qm_busy = {}               # dictionary of {QM ranks: DFT start time}
        
        qm_data_path = os.path.join(mg_path, 'data_to_qm.npy')
        ml_data_path = os.path.join(mg_path, 'data_to_ml.npy')
        test_res_path = os.path.join(mg_path, "testset_results")
        qm_failed_path = os.path.join(mg_path, 'qm_failed.npy')
        qm_failed = []
        
        if os.path.exists(qm_data_path):
            with open(qm_data_path, 'rb') as fh:
                coords_to_qm = list(np.load(fh))
                states_to_qm = list(np.load(fh))
        else:
            coords_to_qm = []      # buffer to store coordinates if no idle QM process
            states_to_qm = []       # buffer to store corresponding states of coordinates in coords_to_qm (dft calculation requires specifying the state)
            
        if os.path.exists(ml_data_path):
            with open(ml_data_path, 'rb') as fh:
                coords_to_ml = list(np.load(fh))
                energy_to_ml = list(np.loat(fh))
                force_to_ml = list(np.load(fh))
        else:
            coords_to_ml = []      # list of coordinate to retrain ML models
            energy_to_ml = []      # list of energy to retrain ML models
            force_to_ml = []       # list of force to retrain ML models
        
        save_progress = 0.0
        time_start = time.time()
        while not stop_run:
            if time.time() - time_start >= update_time:
                # set the save_progress flag to true and reset the starting point time
                save_progress = 1.0
                time_start = time.time()
            else:
                save_progress = 0.0
            
            # non-blocking receive save_progress flag, coordinates and states for QM calculations
            status_ex_mg = MPI.Status()
            if comm_world.Iprobe(source=RANK_EX, tag=t_ex_mg, status=status_ex_mg):
                data_from_ex = np.empty((status_ex_mg.Get_count(MPI.DOUBLE)), dtype=float)
                comm_world.Irecv([data_from_ex, MPI.DOUBLE], source=RANK_EX, tag=t_ex_mg)
                # organize data received from EX progress
                n_coord = int(data_from_ex.shape[0] / (n_atoms*3 + 1))
                coords_to_qm += list(data_from_ex[:n_coord*n_atoms*3].reshape(n_coord, n_atoms, 3))
                states_to_qm += list(data_from_ex[n_coord*n_atoms*3:].reshape(n_coord,))
                print("Data from Exchange have been received.")
                
            # collect QM calculation results for ML training
            to_remove = []
            for i, t in qm_busy.items():
                if time.time() - t >= dft_wait_time:
                    tag_here = t_qm_mg[i - rank_qm[0]]
                    if comm_world.Iprobe(source=i, tag=tag_here):
                        # calculation of QM process i is finished and results have arrived to MG process
                        to_remove.append(i)
                        qm_free.append(i)    # add rank i to QM idle list
                        # receive calculation results from QM process i
                        data_from_qm = np.empty((n_atoms*3 + n_states + n_states*n_atoms*3,), dtype=float)
                        comm_world.Recv([data_from_qm, MPI.DOUBLE], source=i, tag=tag_here)
                        # QM process returns None for force calculation if it fails
                        if (np.nan_to_num(data_from_qm[n_atoms*3+n_states:]) == 0).all():
                            qm_failed.append(data_from_qm[:n_atoms*3].reshape(n_atoms,3))
                            continue
                        coords_to_ml.append(data_from_qm[:n_atoms*3].reshape(n_atoms,3))
                        energy_to_ml.append(data_from_qm[n_atoms*3:n_atoms*3+n_states].reshape(n_states,))
                        force_to_ml.append(data_from_qm[n_atoms*3+n_states:].reshape(n_states,n_atoms,3))
                        print("Data from QM rank {i} has arrived.")
                    else:
                        # calculation results of QM process i have not arrived yet
                        # wait for at least another 2 minutes before checking again
                        qm_busy[i] += 120
            for i in to_remove:
                qm_busy.pop(i)
            
            # send new data to ML process for training
            if len(coords_to_ml) >= retrain_step and (req_ml_data is None or req_ml_data.Test()):
                assert len(coords_to_ml) == len(energy_to_ml) and len(coords_to_ml) == len(force_to_ml), \
                    "lengths of coords_to_ml/energy_to_ml/force_to_ml don't match."
                data_to_ml = np.concatenate((coords_to_ml[:retrain_step].flatten(), energy_to_ml[:retrain_step], force_to_ml[:retrain_step], [float(len(coords_to_qm)),]), axis=0)
                req_ml_data = comm_ml_mg.Ibcast([data_to_ml, MPI.DOUBLE], root=0)
                coords_to_ml = coords_to_ml[retrain_step:]
                energy_to_ml = energy_to_ml[retrain_step:]
                force_to_ml = force_to_ml[retrain_step:]
                print("New data have been broadcasted to ML processes.")
                
                # sort the coords_to_qm and states_to_qm with the most updated ML models
                if len(coords_to_qm) > 1:
                    req_ml_data.Wait()
                    req_ml_data = None
                    del data_to_ml
                    gc.collect()
                    # broadcast coordinates in coords_to_qm to all ML processes
                    comm_ml_mg.Bcast([np.array(coords_to_qm, dtype=float).flatten(), MPI.DOUBLE], root=0)
                    # gather energy predictions from all ML processes
                    pred_to_ml = np.empty((len(coords_to_qm)*n_states,), dtype=float)
                    eng_predictions = np.empty(((model_total+1)*len(coords_to_qm)*n_states,), dtype=float)
                    comm_ml_mg.Gather([pred_to_ml, MPI.DOUBLE], [eng_predictions, MPI.DOUBLE], root=0)
                    # organize energy predictions collected from ML processes
                    eng_predictions = eng_predictions.reshape(model_total+1, len(coords_to_qm), n_states)[1:]
                    # calculate STD for energy predictions, remove coordinates with STD lower than thresh and sort coordinates according to STD values
                    eng_std = np.std(eng_predictions, axis=0, ddof=1)
                    assert eng_std.shape == (len(coords_to_qm), n_states)
                    eng_idx_sorted = np.argsort(np.mean(eng_std, axis=1), axis=0)
                    coords_to_qm = np.array(coords_to_qm, dtype=float)[eng_idx_sorted]
                    states_to_qm = np.array(states_to_qm, dtype=float)[eng_idx_sorted]
                    eng_std = eng_std[eng_idx_sorted]
                    coords_to_qm = list(coords_to_qm[np.nonzero((eng_std > thresh).any(axis=1))[0]])
                    states_to_qm = list(states_to_qm[np.nonzero((eng_std > thresh).any(axis=1))[0]])
                    # free memory
                    del eng_predictions, eng_std
                    gc.collect()
                    print("coords_to_qm and states_to_qm sorted.")
            
            # send coordinates and states to idle QM processes for calculation
            while len(coords_to_qm) > 0 and len(qm_free) > 0:
                qm_free_rank = qm_free.pop()
                data_to_qm = np.append(coords_to_qm.pop(-1).flatten(), [states_to_qm.pop(-1),], axis=0)
                tag_here = t_qm_mg[qm_free_rank - rank_qm[0]]
                comm_world.Send([data_to_qm, MPI.DOUBLE], dest=qm_free_rank, tag=tag_here)
                print(f"Coordinates and current state have been send to QM process {qm_free_rank}")
                qm_busy[qm_free_rank] = time.time()
                
            # save progress
            if save_progress == 1.0:
                # save coordinates where QM calculations fail
                if len(qm_failed) > 0:
                    mode = 'ab' if os.path.exists(qm_failed_path) else 'wb'
                    with open(qm_failed_path, mode) as fh:
                        np.save(fh, np.array(qm_failed, dtype=float))
                    del qm_failed
                    gc.collect()
                    qm_failed = []
                    save_progress = 0.0
                    print("qm_failed saved.")
                # save coords_to_qm and states_to_qm
                if len(coords_to_qm) > 0:
                    with open(qm_data_path, 'wb') as fh:
                        np.save(fh, np.array(coords_to_qm, dtype=float))
                        np.save(fh, np.array(states_to_qm, dtype=float))
                    save_progress = 0.0
                    print("coords_to_qm and states_to_qm saved.")
                # save coords_to_ml, energy_to_ml and force_to_ml
                if len(coords_to_ml) > 0:
                    with open(ml_data_path, 'wb') as fh:
                        np.save(fh, np.array(coords_to_ml, dtype=float))
                        np.save(fh, np.array(energy_to_ml, dtype=float))
                        np.save(fh, np.array(force_to_ml, dtype=float))
                    save_progress = 0.0
                    print("ML data saved.")
                
        print("The End.")
        fstdout.close()
            
    
    
    

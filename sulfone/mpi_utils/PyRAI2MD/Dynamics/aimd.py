######################################################
#
# PyRAI2MD 2 module for ab initio molecular dynamics
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import time, os, pickle, sys, json
import numpy as np
from mpi4py import MPI

from PyRAI2MD.Dynamics.Propagators.surface_hopping import SurfaceHopping
from PyRAI2MD.Dynamics.Ensembles.ensemble import Ensemble
from PyRAI2MD.Dynamics.verlet import VerletI, VerletII
from PyRAI2MD.Dynamics.reset_velocity import ResetVelo
from PyRAI2MD.Utils.timing import WhatIsTime, HowLong
from PyRAI2MD.Utils.coordinates import PrintCoord

class AIMD:
    """ Ab initial molecular dynamics class

        Parameters:          Type:
            trajectory       class       trajectory class
            keywords         dict        keyword dictionary
            qm               class       QM method class
            id               int         trajectory id index
            dir              boolean     create a subdirectory

        Attributes:          Type:
            version          str         version information header
            title            str         calculation title
            maxerr_energy    float       maximum energy error threshold
            maxerr_grad      float       maximum gradient error threshold
            maxerr_nac       float       maximum nac error threshold
            maxerr_soc       float       maximum soc error threshold
            silent           int         silient mode for screen output
            verbose          int         verbose level of output information
            direct           int         number of steps to record output
            buffer           int         number of steps to skip output
            record           int         record the history of trajectory
            checkpoint       int         trajectory checkpoint frequency
            restart          int         restart calculation
            addstep          int         number of steps that will be added in restarted calculation
            stop             int         trajectory termination signal
            skipstep         int         number of steps being skipped to write output
            skiptraj         int         number of steps being skipped to save trajectory

        Functions:           Returns:
            run              class       run molecular dynamics simulation

    """

    def __init__(self, trajectory = None, keywords = None, qm = None, id = None, dir = None, **kwargs):
        self.timing = 0  ## I use this to test calculation time

        ## initialize variables
        self.version       = keywords['version']
        self.title         = keywords['control']['title']
        self.maxerr_energy = keywords['control']['maxenergy']
        self.maxerr_grad   = keywords['control']['maxgrad']
        self.maxerr_nac    = keywords['control']['maxnac']
        self.maxerr_soc    = keywords['control']['maxsoc']
        self.silent        = keywords['md']['silent']
        self.verbose       = keywords['md']['verbose']
        self.direct        = keywords['md']['direct']
        self.buffer        = keywords['md']['buffer']
        self.record        = keywords['md']['record']
        self.checkpoint    = keywords['md']['checkpoint']
        self.restart       = keywords['md']['restart']
        self.addstep       = keywords['md']['addstep']
        self.stop          = 0
        self.skipstep      = 0
        self.skiptraj      = 0
        self.comm_md_mg    = kwargs.get('comm', None)
        self.mpi_status    = 0
        self.bond_index    = kwargs.get('bond_index', None)
        self.bond_limit    = kwargs.get('bond_limit', None)
        self.time_md       = kwargs.get('time_md', None)
        self.traj_data     = kwargs.get('traj_data', None)

        ## update calculation title if the id is available
        if id != None:
            self.title = '%s-%s' % (self.title, id)

        ## setup molecular dynamics calculation path
        if dir != None:
            self.logpath = '%s/%s' % (os.getcwd(), self.title)
            if os.path.exists(self.logpath) == False:
                os.makedirs(self.logpath)
        else:
            self.logpath = os.getcwd()

        ## create a trajectory object
        self.traj = trajectory

        ## create an electronic method object
        self.QM = qm

        ## check if it is a restart calculation and if the previous check point pkl file exists
        if self.restart == 1:
            check_f1 = os.path.exists('%s/%s.pkl'         % (self.logpath, self.title))
            check_f2 = os.path.exists('%s/%s.log'         % (self.logpath, self.title))
            check_f3 = os.path.exists('%s/%s.md.energies' % (self.logpath, self.title))
            check_f4 = os.path.exists('%s/%s.md.xyz'      % (self.logpath, self.title))
            check_f5 = os.path.exists('%s/%s.md.velo'     % (self.logpath, self.title))
            check_f6 = os.path.exists('%s/%s.sh.energies' % (self.logpath, self.title))
            check_f7 = os.path.exists('%s/%s.sh.xyz'      % (self.logpath, self.title))
            check_f8 = os.path.exists('%s/%s.sh.velo'     % (self.logpath, self.title))
            checksignal = int(check_f1)\
                        + int(check_f2)\
                        + int(check_f3)\
                        + int(check_f4)\
                        + int(check_f5)\
                        + int(check_f6)\
                        + int(check_f7)\
                        + int(check_f8)

            if checksignal == 8:
                with open('%s/%s.pkl' % (self.logpath, self.title),'rb') as mdinfo:
                    self.traj = pickle.load(mdinfo)
            else:
                sys.exit("""\n PyRAI2MD: Checkpoint files are incomplete.
  Cannot restart, please consider start it over again, sorry!

                File          Found
                pkl           %s
       	       	log           %s
       	       	md.energies   %s
       	       	md.xyz        %s
       	       	md.velo       %s
       	       	sh.energies   %s
       	       	sh.xyz        %s
                sh.velo       %s
                """ % (check_f1, check_f2, check_f3, check_f4, check_f5, check_f6, check_f7, check_f8))

        ## check if it is a freshly new calculation then create output files
        ## otherwise, the new results will be appended to the existing log in a restart calculation
        elif self.restart == 0 or os.path.exists('%s/%s.log' % (self.logpath, self.title)) == False:
            log = open('%s/%s.log' % (self.logpath, self.title),'w')
            log.close()
            log = open('%s/%s.md.energies' % (self.logpath, self.title),'w')
            log.close()
            log = open('%s/%s.sh.energies' % (self.logpath, self.title),'w')
            log.close()
            log = open('%s/%s.md.xyz' % (self.logpath, self.title),'w')
            log.close()
            log = open('%s/%s.sh.xyz' % (self.logpath, self.title),'w')
            log.close()
            log = open('%s/%s.md.velo' % (self.logpath, self.title),'w')
            log.close()
            log = open('%s/%s.sh.velo' % (self.logpath, self.title),'w')
            log.close()

    def _propagate(self, **kwargs):
        ##
	##-----------------------------------------
        ## ---- Initial Kinetic Energy Scaling
        ##-----------------------------------------
        ##
        if self.traj.iter == 1:
            f = 1
            ## add excess kinetic energy in the first step if requested
            if self.traj.excess != 0:
                K0 = self._kinetic_energy(self.traj)
                f = ((K0 + self.traj.excess) / K0)**0.5

            ## scale kinetic energy in the first step if requested
            if self.traj.scale != 1:
                f = self.traj.scale**0.5

            ## scale kinetic energy to target value in the first step if requested
            if self.traj.target != 0:
                K0 = self._kinetic_energy(self.traj)
                f = (self.traj.target / K0)**0.5

            self.traj.velo *= f

	##-----------------------------------------
        ## ---- Trajectory Propagation
	##-----------------------------------------
        ##
        ## update previous-preivous and previous nuclear properties              
        self.traj.update_nu()

        ## update current kinetic energies, coordinates, and gradient
        self.traj = VerletI(self.traj)
        if self.timing == 1: print('verlet', time.time())
        
        self.traj = self._potential_energies(self.traj)
        ## check if PL predictions passed std test (for mpi interface only)
        if self.mpi_status == 1:
            return None
        if self.timing == 1: print('compute_egn', time.time())

        self.traj = VerletII(self.traj)
        if self.timing == 1: print('verlet_2', time.time())

        self.traj = self._kinetic_energy(self.traj)

        ##
	##-----------------------------------------
        ## ---- Velocity Adjustment
	##-----------------------------------------
        ##
        ## reset velocity to avoid flying ice cube
        ## end function early if velocity reset is not requested
        if self.traj.reset != 1:
            return None

       	## end function early if	velocity reset step is 0 but iteration is more than 1
        if self.traj.resetstep == 0 and self.traj.iter > 1:
            return None

        ## end function early if velocity reset step is not 0 but iteration is not the multiple of it 
        if self.traj.resetstep != 0:
            if self.traj.iter % self.traj.resetstep != 0:
                return None

        ## finally reset velocity here
        self.traj = ResetVelo(self.traj)

    def _potential_energies(self, traj):
        if self.comm_md_mg is None:
            traj = self.QM.evaluate(traj)
        else:
            ## get energy/force through MPI interface
            # communicate through MPI to get energy/force predictions
            coords_send = np.concatenate((np.array([float(traj.state-1),]), traj.coord.flatten()), axis=0)
            coords_collect = None
            # send coordinates to PL process through MG process
            self.t1 = time.time()
            self.comm_md_mg.Gather([coords_send, MPI.DOUBLE], coords_collect, root=0)
            self.t2 = time.time()
            # receive energy and force predictions through MG process
            data_to_md = None
            data_recv = np.empty((traj.nstate+traj.nstate*traj.natom*3+1,), dtype=float)
            self.comm_md_mg.Scatter([data_to_md, MPI.DOUBLE], [data_recv, MPI.DOUBLE], root=0)
            self.t3 = time.time()
            save_data = int(data_recv[-1])
            data_recv = data_recv[:-1]
            eng_recv = data_recv[:traj.nstate].reshape(traj.nstate,) / 27.21138624598853  # eV to Hatree
            grad_recv = data_recv[traj.nstate:].reshape(traj.nstate, traj.natom, 3) / 27.21138624598853 * 0.52917721090380  # eV to Hatree
            
            if self.traj_data['save_data'] < save_data:
                self.traj_data['save_data'] = save_data
            
            nac = []
            soc = []
            err_e = 0
            err_g =	0
            err_n =	0
            err_s =	0
            
            traj.energy = np.copy(eng_recv)
            traj.grad = np.copy(grad_recv)
            traj.nac = np.copy(nac)
            traj.soc = np.copy(soc)
            traj.err_energy = err_e
            traj.err_grad = err_g
            traj.err_nac = err_n
            traj.err_soc = err_s
            traj.status = 1
            
            # check if PL predictions fail the std test
            if eng_recv[0] == 0.0:
                self.mpi_status = 1
            else:
                self.traj_data['energy'][-1].append(eng_recv.tolist())
                self.traj_data['force'][-1].append(grad_recv[traj.state-1].tolist())
                self.traj_data['coord'][-1].append(traj.coord.tolist())
                self.traj_data['state'][-1].append(int(traj.state-1))

        return traj

    def _kinetic_energy(self, traj):
        traj.kinetic = np.sum(0.5 * (traj.mass * traj.velo**2))

        return traj

    def _thermodynamic(self):
        self.traj = Ensemble(self.traj)

        return self.traj

    def _surfacehop(self):
        ## update previous population, energy matrix, and non-adiabatic coupling matrix
        self.traj.update_el()

        # update current population, energy matrix, and non-adiabatic coupling matrix
        self.traj = SurfaceHopping(self.traj)

        return self.traj

    def _reactor(self):
        ## TODO periodically adjust velocity to push reactants toward center

        return self

    def _heading(self):
        state_info = ''.join(['%4d' % (x+1) for x in range(len(self.traj.statemult))])
        mult_info = ''.join(['%4d' % (x) for x in self.traj.statemult])

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |          Nonadiabatic Molecular Dynamics          |
 |                                                   |
 *---------------------------------------------------*


 State order:      %s
 Multiplicity:     %s

 QMMM key:         %s
 QMMM xyz          %s
 Active atoms:     %s
 Inactive atoms:   %s
 Link atoms:       %s
 Highlevel atoms:  %s
 Lowlevel atoms:   %s

""" % ( self.version,
        state_info,
        mult_info,
        self.traj.qmmm_key,
        self.traj.qmmm_xyz,
        self.traj.natom,
        self.traj.ninac,
        self.traj.nlink,
        self.traj.nhigh,
        self.traj.nlow)

        return headline

    def _chkerror(self):
        ## This function check the errors in energy, force, and NAC
        ## This function stop MD if the errors exceed the threshold
        ## This function stop MD if the qm calculation failed

        if self.traj.err_energy != None and\
            self.traj.err_grad != None and\
            self.traj.err_nac != None and\
            self.traj.err_soc != None:
            if  self.traj.err_energy > self.maxerr_energy or\
                self.traj.err_grad > self.maxerr_grad or\
                self.traj.err_nac > self.maxerr_nac or\
                self.traj.err_soc > self.maxerr_soc:
                self.stop = 1
        elif self.traj.status == 0:
            self.stop = 2

    def	_chkpoint(self):
        ## record the last a few MD step
        self.traj.record()

        ## prepare a comment line for xyz file
        cmmt = '%s coord %d state %d' % (self.title, self.traj.iter, self.traj.last_state)

        ## prepare the surface hopping section using Molcas output format
        ## add surface hopping information to xyz comment line
        if   self.traj.hoped == 0:
            hop_info = ' A surface hopping is not allowed\n  **\n At state: %3d\n' % (self.traj.state)
        elif self.traj.hoped == 1:
       	    hop_info = ' A surface hopping event happened\n  **\n From state: %3d to state: %3d *\n' % (
                self.traj.last_state, self.traj.state)
            cmmt += ' to %d CI' % (self.traj.state)
        elif self.traj.hoped == 2:
            hop_info = ' A surface hopping is frustrated\n  **\n At state: %3d\n' % (self.traj.state)

        ## prepare population and potential energy info
        pop = ' '.join(['%28.16f' % (x) for x in np.diag(np.real(self.traj.A))])
        pot = ' '.join(['%28.16f' % (x) for x in self.traj.energy])

        ## prepare xyz, velo, and energy info
        xyz_info = '%d\n%s\n%s' % (self.traj.natom, cmmt, PrintCoord(np.concatenate((self.traj.atoms, self.traj.coord), axis = 1)))
        velo_info = '%d\n%s\n%s' % (self.traj.natom, cmmt, PrintCoord(np.concatenate((self.traj.atoms, self.traj.velo), axis = 1)))
        energy_info = '%20.2f%28.16f%28.16f%28.16f%s\n' % (
            self.traj.iter * self.traj.size,
            self.traj.energy[self.traj.last_state - 1],
            self.traj.kinetic,
            self.traj.energy[self.traj.last_state - 1] + self.traj.kinetic, 
            pot)

        ## prepare logfile info
        log_info = ' Iter: %8d  Ekin = %28.16f au T = %8.2f K dt = %10d CI: %3d\n Root chosen for geometry opt %3d\n' % (
            self.traj.iter, 
            self.traj.kinetic,
            self.traj.temp,
            self.traj.size,
            self.traj.nstate,
            self.traj.last_state)

        log_info += '\n Gnuplot: %s %s %28.16f\n  **\n  **\n  **\n%s\n' % (
            pop,
            pot,
            self.traj.energy[self.traj.last_state - 1],
            hop_info)

        ## add verbose info
        log_info += self._verbose_log_info(self.verbose)

        ## add error info
        if  self.traj.err_energy != None and\
            self.traj.err_grad != None and\
            self.traj.err_nac != None and\
            self.traj.err_soc != None:

            log_info += """
  &surface hopping information
-------------------------------------------------------
%s
-------------------------------------------------------

  &error iter %-10s
-------------------------------------------------------
  Energy   MaxStDev:          %-10.4f
  Gradient MaxStDev:          %-10.4f
  Nac      MaxStDev:          %-10.4f
  Soc      MaxStDev:          %-10.4f
-------------------------------------------------------

""" % ( self.traj.shinfo,
        self.traj.iter,
        self.traj.err_energy,
        self.traj.err_grad,
        self.traj.err_nac,
        self.traj.err_soc)

        ## print log on screen
        if self.silent == 0:
            print(log_info)

        ## always record surface hopping event 
        if self.traj.hoped == 1:
            self._record_surface_hopping(
                self.logpath,
                self.title,
                energy_info,
                xyz_info,
                velo_info)

        ## checkpoint trajectory class to pkl
        if self.checkpoint > 0:
            self.skiptraj = self._step_counter(self.skiptraj, self.checkpoint)
            self.skiptraj = self._force_output(self.skiptraj)
        else:
            self.skiptraj = 1
        if self.skiptraj == 0:
            with open('%s.pkl' % (self.title),'wb') as mdtraj:
                pickle.dump(self.traj, mdtraj)

        ## write logfile to disk
        if self.traj.iter > self.direct:
            self.skipstep = self._step_counter(self.skipstep, self.buffer)
            self.skipstep = self._force_output(self.skipstep)
        if self.skipstep == 0:
            self._dump_to_disk(self.logpath,
                self.title,
                log_info,
                energy_info,
                xyz_info,
                velo_info)

    def _step_counter(self, counter, step):
        counter += 1

        ## reset counter to 0 at printing step or end point or stopped point
        if counter == step or counter == self.traj.step or self.stop == 1:
            counter = 0

        return counter

    def _force_output(self, counter):
        if self.traj.iter == self.traj.step or self.stop != 0:
            counter = 0
        return counter

    def _verbose_log_info(self, verbose):
        log_info = ''

        if verbose == 0:
            return log_info

        log_info += """
  &coordinates in Angstrom
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (PrintCoord(np.concatenate((self.traj.atoms, self.traj.coord), axis = 1)))

        log_info += """
  &velocities in Bohr/au
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (PrintCoord(np.concatenate((self.traj.atoms, self.traj.velo), axis = 1)))

        for n in range(self.traj.nstate):
            try:
                grad = self.traj.grad[n]
                log_info += """
  &gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (n + 1, PrintCoord(np.concatenate((self.traj.atoms, grad), axis = 1)))

            except IndexError:
                log_info += """
  &gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
  Not Computed
-------------------------------------------------------------------------------
""" % (n + 1)

        for n, pair in enumerate(self.traj.nac_coupling):
            s1, s2 = pair
            m1 = self.traj.statemult[s1]
            m2 = self.traj.statemult[s2]
            try:
                coupling = self.traj.nac[n]
                log_info += """
  &nonadibatic coupling %3d - %3d in Hartree/Bohr M = %1d / %1d
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (s1 + 1, s2 + 1, m1, m2, PrintCoord(np.concatenate((self.traj.atoms, coupling), axis = 1)))

            except IndexError:
                log_info += """
  &nonadibatic coupling %3d - %3d in Hartree/Bohr M = %1d / %1d
-------------------------------------------------------------------------------
  Not computed
-------------------------------------------------------------------------------
""" % (s1 + 1, s2 + 1, m1, m2)

        soc_info = ''
        for n, pair in enumerate(self.traj.soc_coupling):
            s1, s2 = pair
            m1 = self.traj.statemult[s1]
            m2 = self.traj.statemult[s2]
            try:
                coupling = self.traj.soc[n]
                soc_info += '  <H>=%10.4f            %3d - %3d in cm-1 M1 = %1d M2 = %1d\n' % (
                    coupling, s1 + 1, s2 + 1, m1, m2)

            except:
                soc_info += '  Not computed              %3d - %3d in cm-1 M1 = %1d M2 = %1d\n' % (
                    s1 + 1, s2 + 1, m1, m2)

        if len(self.traj.soc_coupling) > 0:
            log_info += """
  &spin-orbit coupling
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (soc_info)

        return log_info

    def _dump_to_disk(self, logpath, title, log_info, energy_info, xyz_info, velo_info):
        ## output data to disk
        with open('%s/%s.log' % (logpath, title), 'a') as log:
            log.write(log_info)

        with open('%s/%s.md.energies' % (logpath, title), 'a') as log:
            log.write(energy_info)

        with open('%s/%s.md.xyz' % (logpath, title), 'a') as log:
            log.write(xyz_info)

        with open('%s/%s.md.velo' % (logpath, title), 'a') as log:
            log.write(velo_info)

    def _record_surface_hopping(self, logpath, title, energy_info, xyz_info, velo_info):
        ## output data for surface hopping event to disk
        with open('%s/%s.sh.energies' % (logpath,title),'a') as log:
            log.write(energy_info)

        with open('%s/%s.sh.xyz' % (logpath,title),'a') as log:
            log.write(xyz_info)

        with open('%s/%s.sh.velo' % (logpath,title),'a') as log:
            log.write(velo_info)
            
    def _calc_bond_length(self, coord):
        c1 = np.expand_dims(coord, axis=0)
        c2 = np.expand_dims(coord, axis=1)
        dist_matrix = np.sqrt(np.sum(np.square(c1-c2), axis=-1))
        bond_length = {}
        for b, idx in self.bond_index.items():
            bond_length[b] = [dist_matrix[i,j] for (i,j) in idx]
        return bond_length
    
    def _check_bond_length(self, traj):
        bond_lengths = self._calc_bond_length(traj.coord)
        for b, l in self.bond_limit.items():
            if (np.array(bond_lengths[b], dtype=float) > l).any():
                return False, b
        return True, None

    def run(self):
        warning  = ''
        start = time.time()

        ## add heading to new output files
        if self.restart == 0:
            heading = 'Nonadiabatic Molecular Dynamics Start: %20s\n%s' % (WhatIsTime(), self._heading())
            with open('%s/%s.log' % (self.logpath, self.title), 'a') as log:
                log.write(heading)

            mdhead = '%20s%28s%28s%28s%28s\n' % ('time', 'Epot', 'Ekin', 'Etot', 'Epot1,2,3...')
            with open('%s/%s.md.energies' % (self.logpath, self.title),'a') as log:
                log.write(mdhead)

            with open('%s/%s.sh.energies' % (self.logpath, self.title),'a') as log:
                log.write(mdhead)

        if self.silent == 0:
            print(heading)

        ## loop over molecular dynamics steps
        self.traj.step += self.addstep
        for iter in range(self.traj.step - self.traj.iter):
            self.t0 = time.time()
            self.traj.iter += 1
            if self.timing == 1: print('start', time.time())

            ## propagate nuclear positions (E,G,N,R,V,Ekin)
            self._propagate()
            if self.timing == 1: print('propagate', time.time())
            ## check if PL predictions passed std test (for mpi interface only)
            if self.mpi_status == 1:
                self.t4 = time.time()
                if self.time_md != None:
                    self.time_md['gather'].append(self.t2 - self.t1)
                    self.time_md['scatter'].append(self.t3 - self.t2)
                    self.time_md['prop'].append(self.t4 -self.t0 - self.t3 + self.t1)
                warning = 'Trajectory terminated beacuse std of PL predictions is too high.'
                break

            ## adjust kinetics (Ekin,V,thermostat)
            self._thermodynamic()
            if self.timing == 1: print('thermostat', time.time())

            ## detect surface hopping
            self._surfacehop()   # update A,H,D,V,state
            if self.timing == 1: print('surfacehop', time.time())

            ## check errors and checkpointing
            self._chkerror()
            self._chkpoint()
            if self.timing == 1: print('save', time.time())
            
            ## check if bond length has exceeded limit
            normal_bond_length, b = self._check_bond_length(self.traj)
            #if not normal_bond_length:
            #    warning = f'Trajectory terminated beacuse bond {b} has exceeded bond length limit.'
            #    break
            
            self.t4 = time.time()
            if self.time_md != None:
                self.time_md['gather'].append(self.t2 - self.t1)
                self.time_md['scatter'].append(self.t3 - self.t2)
                self.time_md['prop'].append(self.t4 -self.t0 - self.t3 + self.t1)
                #with open('mdtime.json', 'w') as fh:
                #    json.dump(self.time_md, fh)
            
            ## terminate trajectory
            if not normal_bond_length:
                warning = f'Trajectory terminated beacuse bond {b} has exceeded bond length limit.'
                break
            if   self.stop == 1:
                warning = 'Trajectory terminated because the NN prediction differences are larger than thresholds.'
                break
            elif self.stop == 2:
                warning = 'Trajectory terminated because the QM calculation failed.'
                break

        end = time.time()
        walltime = HowLong(start, end)
        tailing = '%s\nNonadiabatic Molecular Dynamics End: %20s Total: %20s\n' % (warning, WhatIsTime(), walltime)

        if self.silent == 0:
            print(tailing)

        with open('%s/%s.log' % (self.logpath, self.title),'a') as log:
            log.write(tailing)
            
        if self.traj.iter < self.traj.step and len(self.traj_data['energy'][-1]) > 0:
            engs_traj = np.array(self.traj_data['energy'][-1], dtype=float)
            engs_traj = np.append(engs_traj, np.zeros((self.traj.step-engs_traj.shape[0], engs_traj.shape[1]), dtype=float), axis=0)
            self.traj_data['energy'][-1] = engs_traj.tolist()
            force_traj = np.array(self.traj_data['force'][-1], dtype=float)
            force_traj = np.append(force_traj, np.zeros((self.traj.step-force_traj.shape[0], force_traj.shape[1], force_traj.shape[2]), dtype=float), axis=0)
            self.traj_data['force'][-1] = force_traj.tolist()
            coord_traj = np.array(self.traj_data['coord'][-1], dtype=float)
            coord_traj = np.append(coord_traj, np.zeros((self.traj.step-coord_traj.shape[0], coord_traj.shape[1], coord_traj.shape[2]), dtype=float), axis=0)
            self.traj_data['coord'][-1] = coord_traj.tolist()
            state_traj = np.array(self.traj_data['state'][-1], dtype=int)
            state_traj = np.append(state_traj, np.zeros((self.traj.step-state_traj.shape[0]), dtype=int), axis=0)
            self.traj_data['state'][-1] = state_traj.tolist()
        elif len(self.traj_data['energy'][-1]) == 0:
            self.traj_data['energy'][-1] = np.zeros((self.traj.step, self.traj.nstate), dtype=float).tolist()
            self.traj_data['force'][-1] = np.zeros((self.traj.step, self.traj.natom, 3), dtype=float).tolist()
            self.traj_data['coord'][-1] = np.zeros((self.traj.step, self.traj.natom, 3), dtype=float).tolist()
            self.traj_data['state'][-1] = np.zeros((self.traj.step,), dtype=float).tolist()
        return self.traj

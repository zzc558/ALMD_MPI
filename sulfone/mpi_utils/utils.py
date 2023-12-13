from __future__ import print_function
from __future__ import absolute_import
import shutil
import uuid
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as scsp
import time
import subprocess
import shlex
import xtb_utils as xtb


HToEV=27.21138505
AToBohr=1.889725989

hbar_eVs=6.582e-16 # eV
e=1.6e-19 # C
m_to_A=1e-10
c = 2.998e8 # m/s
NA = 6.022e23 # g/mol

elementmasses={"C":12.01115,
        "H":1.00797,
        "O":15.99940,
        "N":14.007
        }



def sampling(settings, coords, elements, hess, num = 100):

    reduced_masses_np = settings["reduced_masses"]
    
    wavenumbers_np = settings["vibspectrum"][-len(reduced_masses_np):] ## 1/cm


    reduced_masses_g = reduced_masses_np/NA # g
    reduced_masses_kg = reduced_masses_g*0.001 # kg

    forceconstants = 4.0 * np.pi**2 * c**2 * wavenumbers_np**2 *100**2.0 * reduced_masses_kg ## N/m or J/m^2
    forceconstants = forceconstants/e*m_to_A**2.0 ## eV / A^2


    n = settings["n"]
    vectors = hess[-len(reduced_masses_np):]
    #print(vectors[0])
    #vectors = vectors.reshape(3*n-6,n*3)
    #print(vectors[0])
    #exit()
    # generate mass weighted vectors that are orthogonal
    mass_per_atom_vector=[]
    for element in elements:
        for i in range(0,3):
            mass_per_atom_vector.append(elementmasses[element])
    mass_per_atom_vector=np.array(mass_per_atom_vector)
    vectors_massweighted=np.zeros((len(vectors),len(coords),3))
    for vec_idx,vec in enumerate(vectors):
        vectors_massweighted[vec_idx]=(vec.flatten()*mass_per_atom_vector**0.5).reshape((len(coords),3))
    
    scalar_products_massweighted=np.zeros((len(vectors),len(vectors)))
    for idx1 in range(len(vectors)):
        for idx2 in range(len(vectors)):
            scalar_products_massweighted[idx1][idx2]=np.sum(vectors_massweighted[idx1].flatten()*vectors_massweighted[idx2].flatten())
            #scalar_products_massweighted[idx1][idx2]=np.sum(vectors[idx1].flatten()*vectors[idx2].flatten())
    #print([scalar_products_massweighted[i][i] for i in range(len(scalar_products_massweighted))])
    #exit()

    # generate orthonormal vectors: they were used by ANI authors
    vectors_orthonormal=np.zeros((len(vectors),len(coords),3))
    for vec_idx,vec in enumerate(vectors_massweighted):
        vectors_orthonormal[vec_idx]=np.copy(vec)/np.linalg.norm(vec.flatten())

    # some parameters from the paper
    Nf=len(wavenumbers_np)
    Na=float(len(coords))
    T = 400.0
    kBT_eV=0.025/300.0*T

    # the non-stochastic part of the coefficients
    Rs0=np.sqrt((3.0*Na*kBT_eV)/(forceconstants))


    coords_init=[]
    elements_init=[]
    n = settings["n"]
    for idx in range(num):
        clash = True
        while clash:
            old=False
            if old:
                c_here = np.copy(coords)
                for v_idx in range(6,3*n):
                    r = np.random.randn()*0.1
                    c_here += r*hess[v_idx]

            else:
                # get random numbers with sum 1
                cs=get_cs(Nf, n)
                # get random signs
                signs=(np.random.randint(0,2,size=Nf)*2-1)
                # get the coefficients
                Rs=signs*Rs0*np.sqrt(cs)#/2.0**0.5
                # calculate the coordinates of the new conformer
                c_here=np.copy(coords)
                for R_idx,R in enumerate(Rs):
                    c_here+=R*np.copy(vectors_orthonormal[R_idx])

            ds = np.sort(scsp.distance.cdist(c_here,c_here).flatten())[n:]
            if min(ds)>0.07:
                clash = False
        coords_init.append(c_here)
        elements_init.append(elements)

    return(coords_init, elements_init)

def get_cs(n_frequencies, n_atoms):
    # sequential generation of random numbers
    cs=np.zeros((n_frequencies))
    s=0.0
    order=np.array(range(n_frequencies))
    np.random.shuffle(order)
    #c_max=1.2
    cs_sum=np.random.random()*(float(int(n_atoms)))**0.5#*2.0#**0.5
    for idx in order:
        c_new=100.0
        while c_new>cs_sum:
            c_new=np.abs(np.random.normal(scale=1.0))/float(n_frequencies)
        cs[idx]=c_new*(cs_sum-s)#np.exp(-1.0/(1.0-s)))#0.5*(1.0-s))
        s=np.sum(cs)
    cs=np.abs(cs)
    return(cs)

def do_sampling(settings, name, num):
    outdir = settings["outdir"]
    if os.path.exists("%s/%s.xyz"%(outdir, name)) and not settings["overwrite"]:
        coords_sampled, elements_sampled = xtb.readXYZs("%s/%s.xyz"%(outdir, name))
        num = len(coords_sampled)
        print("   ---   load %i %s points"%(num, name))
    else:
        print("   ---   generate %i %s points"%(num, name))
        coords_sampled, elements_sampled = sampling(settings, settings["coords"], settings["elements"], settings["hess"], num)
        xtb.exportXYZs(coords_sampled, elements_sampled, "%s/%s.xyz"%(outdir, name))
    return(coords_sampled, elements_sampled)



def vibrations(settings, vib = 0):
    outdir = settings["outdir"]
    if os.path.exists("%s/vib.xyz"%(outdir)) and not settings["overwrite"]:
        print("   ---   load vibration mode")
        coords_vib, elements_vib = xtb.readXYZs("%s/vib.xyz"%(outdir))
    else:
        print("   ---   test vibration mode")
        coords_vib=[]
        elements_vib=[]
        for idx in range(-100,100):
            coords_vib.append(settings["coords"]+idx/100*settings["hess"][vib])
            elements_vib.append(settings["elements"])
        xtb.exportXYZs(coords_vib, elements_vib, "%s/vib.xyz"%(outdir))




def prep_dirs(settings):
    outdir = "output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists("%s/models"%(outdir)):
        os.makedirs("%s/models"%(outdir))
    outdir_test = "output_test"
    if not os.path.exists(outdir_test):
        os.makedirs(outdir_test)
    settings["outdir"] = outdir
    settings["outdir_test"] = outdir_test
    return(outdir, outdir_test)





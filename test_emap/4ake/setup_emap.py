#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from meld.remd import ladder, adaptor, leader
from meld import comm, vault
from meld import system
from meld import parse
import meld.system.montecarlo as mc
from meld.system.restraints import LinearRamp,ConstantRamp
from collections import namedtuple
import glob as glob 
import mrcfile
import copy
import mdtraj as md


N_REPLICAS = 8
N_STEPS = 2000
BLOCK_SIZE = 20


def _get_secondary_sequence(filename=None, contents=None, file=None):
    contents = parse._handle_arguments(filename, contents, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith('#')]
    sequence = ''.join(lines)
    for ss in sequence:
        if not ss in 'HE.':
            raise RuntimeError('Unknown secondary structure type "{}"'.format(aa))
    return sequence


def gen_state(s, index):
    pos = s._coordinates
    pos = pos - np.mean(pos, axis=0)
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    s._box_vectors=np.array([0.,0.,0.])
    energy = 0
    return system.SystemState(pos, vel, alpha, energy,s._box_vectors)

def map_potential(emap, threshold, scale_factor):
    emap_cp = copy.deepcopy(emap)
    emap = scale_factor*((emap - threshold)/(emap.max() - threshold))
    emap_where=np.where(emap<=0)
    emap_cp = scale_factor*(1-(emap_cp - threshold)/(emap_cp.max() - threshold))
    emap_cp[emap_where[0],emap_where[1],emap_where[2]] = scale_factor  
    return emap_cp

def setup_system():
    # load the sequence
#    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
#    n_res = len(sequence.split())
    templates = glob.glob('./1ake_centered.pdb')
                                  
    # build the system
    #p = system.ProteinMoleculeFromSequence(sequence)
    p = system.ProteinMoleculeFromPdbFile(templates[0])
    b = system.SystemBuilder(forcefield="ff14sbside")
    s = b.build_system_from_molecules([p])
    s.temperature_scaler = system.ConstantTemperatureScaler(300.)#
    n_res = s.residue_numbers[-1]
#    s.temperature_scaler = system.GeometricTemperatureScaler(0, 0.4, 300., 450.)
    #
    # Secondary Structure
    #
#    ss_scaler = s.restraints.create_scaler('constant')
#    dist_scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
#    ss_rests = parse.get_secondary_structure_restraints(filename='ss.dat', system=s,ramp=LinearRamp(0,100,0,1), scaler=dist_scaler,
#            torsion_force_constant=2.5, distance_force_constant=2.5)
#    print(ss_rests)
#    n_ss_keep = int(len(ss_rests) * 0.8) #We enforce 100% of restrains 
#    s.restraints.add_selectively_active_collection(ss_rests, n_ss_keep)
    
    map_file=mrcfile.open('4ake_t_5A.mrc')
    map_pot = map_potential(map_file.data,0.25,0.3) 
    map_origin = [float(i)/10 for i in map_file.header['origin'].item()]
    map_origin = [float(i) - np.average(md.load_pdb('1ake_leap.pdb').xyz.T,axis=1)[index] for index,i in enumerate(map_origin)]
    map_voxel = [float(i)/10 for i in map_file.voxel_size.item()]
    map_x = np.linspace(map_origin[0],map_origin[0]+(map_file.data.shape[2]-1)*map_voxel[0],int(map_file.data.shape[2]))
    map_y = np.linspace(map_origin[1],map_origin[1]+(map_file.data.shape[1]-1)*map_voxel[1],int(map_file.data.shape[1]))
    map_z = np.linspace(map_origin[2],map_origin[2]+(map_file.data.shape[0]-1)*map_voxel[2],int(map_file.data.shape[0]))
    map_x,map_y,map_z = np.meshgrid(map_z,map_y,map_x,indexing='ij')
    map_x = np.matrix.flatten(map_x).astype(np.float64)        
    map_y = np.matrix.flatten(map_y).astype(np.float64)
    map_z = np.matrix.flatten(map_z).astype(np.float64)

    print('map_voxel: ',map_voxel)
    print('map_origin: ', map_origin) 
    map_scaler = s.restraints.create_scaler('linear', alpha_min=0.0, alpha_max=1,strength_at_alpha_max=0.93)
    map_res = []
    all_emap_atoms=['N']*n_res+['C']*n_res+['O']*n_res+['CA']*n_res
    map_res.append(s.restraints.create_restraint('emap',map_scaler,atom_res_index=list(range(1,n_res+1))*4,atom_name=all_emap_atoms, mu=np.matrix.flatten(map_pot).astype(np.float64),bandwidth=np.matrix.flatten(np.ones((map_file.data.shape))).astype(np.float64)*map_voxel[0]*0.4,gridpos=np.array([map_z,map_y,map_x]).T))
    s.restraints.add_as_always_active_list(map_res)    
    print(map_file.data.shape)
    dssp=md.compute_dssp(md.load_pdb('1ake_leap.pdb')) 
    E=np.where(dssp=='E')
    H=np.where(dssp=='H')
    beta=[]
    alpha=[]
    tmp=[]
    for i in range(dssp.shape[1]):
        #tmp=[]
        if i in E[1] and i+1 in E[1]:
            tmp.append(i)
        else:
            if len(tmp) >=1:
                beta.append(tmp)
            tmp=[]
    tmp=[] 
    for i in range(dssp.shape[1]):
    #tmp=[]
        if i in H[1] and i+1 in H[1]:
            tmp.append(i)
        else:
            if len(tmp) >=1:
                alpha.append(tmp)
            tmp=[]
    HB=np.concatenate((np.concatenate(beta),np.concatenate(alpha))) 
    torsion_rests = []
    dist_scaler = s.restraints.create_scaler('constant')#, alpha_min=0.4, alpha_max=1.0, factor=4.0)
    psi=np.round(md.compute_psi(md.load_pdb('1ake_leap.pdb'))[1][0]*180/np.pi,2)
    for i in range(1,n_res):
        if i in HB:
            psi_avg = float(psi[i-1])
            psi_sd = 15
            res = i
            psi_rest = s.restraints.create_restraint('torsion', dist_scaler,
                                                      phi=psi_avg, delta_phi=psi_sd, k=0.1,
                                                      atom_1_res_index=res, atom_1_name='N',
                                                      atom_2_res_index=res, atom_2_name='CA',
                                                      atom_3_res_index=res, atom_3_name='C',
                                                      atom_4_res_index=res+1, atom_4_name='N')
            torsion_rests.append(psi_rest)  
    phi=np.round(md.compute_phi(md.load_pdb('1ake_leap.pdb'))[1][0]*180/np.pi,2)
    for i in range(2,n_res+1):            
        if i in HB:
            phi_avg = float(phi[i-2])                
            phi_sd = 15    
            res = i                 
            phi_rest = s.restraints.create_restraint('torsion', dist_scaler,
                                             phi=phi_avg, delta_phi=phi_sd, k=0.1,
                                             atom_1_res_index=res-1, atom_1_name='C',
                                             atom_2_res_index=res, atom_2_name='N',
                                             atom_3_res_index=res, atom_3_name='CA',                    
                                             atom_4_res_index=res, atom_4_name='C')
            torsion_rests.append(phi_rest)  
    print(psi,phi)
    n_tors_keep = int(0.8 * len(torsion_rests)) 
#    s.restraints.add_as_always_active_list(torsion_rests)   
    s.restraints.add_selectively_active_collection(torsion_rests,n_tors_keep)
    rest_group = []     
    pdbs =md.load_pdb('1ake_leap.pdb')
    cas=pdbs.top.select("name CA")
    for i in range(n_res-3):
      if i in HB and i+1 in HB and i+2 in HB: 
        ca_0=pdbs.top.select(f"resid {i} and name CA")
        ca_1=pdbs.top.select(f"resid {i+2} and name CA")
        dist=float(md.compute_distances(pdbs,np.array([ca_0,ca_1]).T)[0][0])
                       
        r1 = dist -0.1 
        if r1 < 0:     
            r1 = 0.0   
        print(r1)
        rest = s.restraints.create_restraint('distance', dist_scaler,LinearRamp(0,100,0,1),
                                              r1=r1, r2=dist, r3=dist+0.1, r4=dist+0.2, k=700,
                                              atom_1_res_index=i+1, atom_2_res_index=i+3,          
                                              atom_1_name='CA', atom_2_name='CA')
        rest_group.append(rest)      
     
    s.restraints.add_as_always_active_list(rest_group)
                    
                    
    # setup mcmc at startup
    # create the options
    options = system.RunOptions()
    options.implicit_solvent_model = 'gbNeck2'
    options.use_big_timestep = False
    options.use_bigger_timestep = True
    options.cutoff = 1.8

    options.use_amap = False
    options.amap_alpha_bias = 1.0
    options.amap_beta_bias = 1.0
    options.timesteps = 222
#    options.min_mc=1
    options.minimize_steps = 100
   # for i in range(30):
   #     print("Heads up! using MC minimizer!")
#    options.min_mc = sched

    # create a store
    store = vault.DataStore(s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)


    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=100)
    policy = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)

    remd_runner = leader.LeaderReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)
    
    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS)
    store.save_communicator(c)
    
    # create and save the initial states
    states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

    return s.n_atoms


setup_system()

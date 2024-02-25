"""
Module of functions to support motion energy analysis of RGC and 
amacrine waves. Similar to Frechette et al (2005), a motion energy 
calculation is performed on various axes of motion.
"""
import sys
sys.path.insert(0,'/home/agogliet/gogliettino/projects/nsem/repos/waves/')
import src.config as cfg
import numpy as np
from scipy import linalg
import pdb
from scipy import stats
import torch

def is_sig(null_distribution,value,alpha=cfg.ALPHA):
    kde = stats.gaussian_kde(null_distribution)
    p = kde.integrate_box_1d(-np.inf,-value) +\
            kde.integrate_box_1d(value,np.inf)
    return p < alpha

def get_shuffled_vectors(
            thetas,
            motion_energies,
            n_sim=cfg.N_SIM,
            seed=cfg.SEED):

    np.random.seed(seed)
    shuffled_vectors = []
    motion_energies_shuffle = motion_energies.copy()

    for i in range(n_sim):
        np.random.shuffle(motion_energies_shuffle)
        vec_sum = compute_vec_sum_polar(thetas[1:],motion_energies_shuffle[1:])
        shuffled_vectors.append(vec_sum)
    
    return np.asarray(shuffled_vectors)

def compute_me_triggered_avg(stimulus,vectors,weighted=False):
    meta_dict = dict()
    meta_dict['global'] = np.mean(stimulus,axis=0)

    # Compute the unweighted average.
    thetas = vectors[:,1] # rows are r,theta pairs
    r = vectors[:,0]

    for direction in cfg.DIRECTIONS:
        dir_f = cfg.DIRECTIONS[direction]
        inds = dir_f(thetas)
        meta_dict[direction] = np.mean(stimulus[inds,...],axis=0)

        if weighted:
            weights = r[inds]
            weights /= np.sum(weights)
            weights = weights[:,None,None]
            meta_dict[direction + '_weighted'] = np.sum(
                                                  stimulus[inds,...] * weights,
                                                  axis=0
                                                 )
    return meta_dict

def compute_motion_energy_torch(theta,speeds,rfs,smoothed_spiketrains,device):
    x = np.cos(theta * (np.pi / 180))
    y = np.sin(theta * (np.pi / 180))
    v = np.array([x,y])
    positions = np.dot(rfs,v)
    m_list = []
    smoothed_spiketrains = torch.tensor(
                                smoothed_spiketrains
                           ).to(torch.float32).to(device)

    for s in speeds:
        dt = positions / s
        shifted_spiketrain_r = torch.tensor(np.asarray([
                                torch.roll(smoothed_spiketrains[i,:],
                                          int(-dt[i])).detach().cpu().numpy()
                                for i in range(smoothed_spiketrains.shape[0])
                               ])).to(torch.float32).to(device)
        shifted_spiketrain_l = torch.tensor(np.asarray([
                                torch.roll(smoothed_spiketrains[i,:],
                                          int(dt[i])).detach().cpu().numpy()
                                for i in range(smoothed_spiketrains.shape[0])
                               ])).to(torch.float32).to(device)
        r = torch.sum(torch.sum(shifted_spiketrain_r,dim=0)**2).item()
        l = torch.sum(torch.sum(shifted_spiketrain_l,dim=0)**2).item()
        m_list.append(r - l)
    
    return m_list
    
def compute_motion_energy_fast(theta,speeds,rfs,smoothed_spiketrains):
    x = np.cos(theta * (np.pi / 180))
    y = np.sin(theta * (np.pi / 180))
    v = np.array([x,y])
    positions = np.dot(rfs,v)
    m_list = []

    for s in speeds:
        dt = positions / s
        shifted_spiketrain_r = np.asarray([
                                np.roll(
                                    smoothed_spiketrains[i,:],int(-dt[i])
                                )
                                for i in range(smoothed_spiketrains.shape[0])
                            ])
        shifted_spiketrain_l = np.asarray([
                                np.roll(
                                    smoothed_spiketrains[i,:],int(dt[i])
                                )
                                for i in range(smoothed_spiketrains.shape[0])
                            ])
        r = np.sum(np.sum(shifted_spiketrain_r,axis=0)**2)
        l = np.sum(np.sum(shifted_spiketrain_l,axis=0)**2)
        m_list.append(r - l)
    
    return m_list

def compute_motion_energy(theta,speeds,rfs,smoothed_spiketrains):
    
    """
    For a given axis of motion and a list of speeds, the motion 
    energy at each speed is computed. Then, the maximum energy
    is computed.
    """
    
    x = np.cos(theta * (np.pi / 180))
    y = np.sin(theta * (np.pi / 180))
    v = np.array([x,y])
    positions = np.dot(rfs,v)
    pos_inds = np.argsort(positions)
    m_list = []

    for s in speeds:
        m = 0

        for k in np.arange(0,pos_inds.shape[0]):
            for q in np.arange(k+1,pos_inds.shape[0]):

                # Compute the pairwise rightward and leftward energy.
                dx = linalg.norm(
                        positions[pos_inds[k]] -\
                        positions[pos_inds[q]]
                    )

                dt = dx / s
                r = np.dot(smoothed_spiketrains[pos_inds[k],:],
                           np.roll(smoothed_spiketrains[pos_inds[q],:],
                                   -np.int(dt)))**2
                l = np.dot(smoothed_spiketrains[pos_inds[k],:],
                           np.roll(smoothed_spiketrains[pos_inds[q],:],
                                   np.int(dt)))**2
                m += (r - l)
        
        m_list.append(m)
        
    return m_list

def get_direction(theta):
    if theta > 135 and theta < 225: return 'left'
    if (theta < 45 and theta > 0) or theta > 315: return 'right'
    if theta > 45 and theta < 135: return 'up'
    if theta > 225 and theta < 315: return 'down'

    assert False, "unexpected value for theta"

def get_max_motion_energies(motion_energy_dict):
    motion_energies = []

    for theta in sorted(list(motion_energy_dict.keys())):
        motion_energies.append(np.max(motion_energy_dict[theta]))
    
    return np.maximum(np.asarray(motion_energies),0)

def compute_vec_sum_polar(thetas,motion_energies):
    """ Computes the vector sum of a bunch of r,Θ pairs """
    cartesian = np.array([np.abs(motion_energies) * np.cos(thetas * (np.pi / 180)),
                      np.abs(motion_energies) * np.sin(thetas * (np.pi / 180))])
    vec_sum = np.sum(cartesian,axis=1)
    vec_sum_polar = np.array([np.sqrt(vec_sum[0]**2+ vec_sum[1]**2),
                              np.arctan2(vec_sum[1],
                                         vec_sum[0]) * (180 / np.pi)])
    vec_sum_polar[1] = (vec_sum_polar[1] + 360) % 360
    
    return vec_sum_polar

def polar_to_cartesian(r,theta):
    """Pass theta in degrees; converts polar coords to cartesian."""
    return np.array([r * np.cos(theta * (np.pi / 180)),
                     r * np.sin(theta * (np.pi / 180))])

def cartesian_to_polar(x,y):
    """Returns theta in degrees; converts cartesian coords to polar"""
    polar = np.array([np.sqrt(x**2+ y**2),np.arctan2(y,x) * (180 / np.pi)])
    polar[1] = (polar[1] + 360) % 360
    
    return polar
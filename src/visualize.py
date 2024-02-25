"""
Tools to aid in the visualization of waves.
"""

import numpy as np
import sys
sys.path.insert(0,'/home/agogliet/gogliettino/projects/nsem/repos/waves/')
import matplotlib.pyplot as plt
import src.config as cfg
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation, PillowWriter

def plot_wave(cellids_dict,smoothed_spikes,ns_tensor):

    """
    Plots the firing rate of ON parasol, OFF parasol, ON midget, OFF midget 
    and A1 amacrine cells during a natural scenes movie.
    """
    n_row = 2
    n_col = 3
    fig,ax = plt.subplots(n_row,n_col,figsize=(15*1.5,5*1.5))
    ax[-1,-1].axis('off')
    handles = []
    j = 0
    k = 0

    for cc,celltype in enumerate(cfg.CELLTYPES_OF_INT):
        cell_inds = cellids_dict['ns_cell_inds_by_celltype'][celltype]
        rfs = cellids_dict['rfs_for_vis'][cell_inds,:]
        handles.append(ax[j,k].scatter(rfs[:,0],rfs[:,1],
                       c=cfg.COLORS[celltype],s=0))
        ax[j,k].axis('off')
        ax[j,k].set_title(cfg.TITLES[celltype])
        k +=1

        if k == n_col:
            k = 0
            j +=1

    def animate(i):
        movie_frame = ns_tensor[...,i]
        j = 0
        k = 0

        for cc,celltype in enumerate(cfg.CELLTYPES_OF_INT):
            cell_inds = cellids_dict['ns_cell_inds_by_celltype'][celltype]
            firing_rates = smoothed_spikes[cell_inds,...]
            firing_rate_frame = firing_rates[:,i]
            ax[j,k].imshow(movie_frame,cmap='gray',
                          vmin=np.min(ns_tensor),vmax=np.max(ns_tensor))
            handles[cc].set_sizes(firing_rate_frame * cfg.SIZE_SCALAR)
            k +=1

            if k == n_col:
                k = 0
                j +=1
            
    anim = animation.FuncAnimation(fig, animate, frames=cfg.N_FRAMES,
                                   interval=cfg.PLAYBACK_INTERVAL)
    plt.close()

    return anim
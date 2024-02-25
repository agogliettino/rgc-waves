"""
Script to preprocess data set. Maps the cellids, bins spike train and smooths
with a kernel.
"""
import sys
sys.path.insert(0,'/home/agogliet/gogliettino/projects/nsem/repos/waves/')
import numpy as np
import visionloader as vl
import os
import src.config as cfg
import src.preprocess as pp
import src.io_util as io
import pdb

ns_datapath_dict = np.load(
                    '../tmp/ns_datapath_dict.npy',
                    allow_pickle=True
                   ).item()

for movie in ns_datapath_dict:

    #if movie not in ['INbrownian_3600_B_045.rawMovie']:
    #    continue
    #if movie not in ['ImageNet_0.rawMovie']:
    #    continue
    if movie not in ['NSbrownian_3000_A_025_fixed.rawMovie']:
        continue

    for dataset in ns_datapath_dict[movie]:

        #if dataset not in ['2017-11-20-4']:
        #    continue
        #if dataset not in ['2017-08-14-1']:
        #    continue
        if dataset not in ['2017-03-15-1']:
            continue
            
        # Map cellids, bin and smooth spiketrain.
        wn_datarun = ns_datapath_dict[movie][dataset]['wn_datarun']
        ns_datarun = ns_datapath_dict[movie][dataset]['ns_datarun']
        frames_per_movie = ns_datapath_dict[movie][dataset]['frames_per_movie']

        if 'frames_grey' in ns_datapath_dict[movie][dataset]:
            frames_per_movie += ns_datapath_dict[movie][dataset]['frames_grey']

        wn_datapath = os.path.join(
                            cfg.PARENT_ANALYSIS,
                            dataset,
                            wn_datarun
                      )
        ns_datapath = os.path.join(
                            cfg.PARENT_ANALYSIS,
                            dataset,
                            ns_datarun
                      )
        wn_vcd = vl.load_vision_data(
                        wn_datapath,
                        os.path.basename(wn_datarun),
                        include_neurons=True,
                        include_ei=True,
                        include_runtimemovie_params=True,
                        include_noise=True,
                        include_params=True
                )
        ns_vcd = vl.load_vision_data(
                        ns_datapath,
                        os.path.basename(ns_datarun),
                        include_neurons=True,
                        include_ei=True,
                        include_noise=True
                )

        # Get the celltypes dict and map cells
        class_path = os.path.join(
                       wn_datapath,
                       cfg.CLASSIFICATION_BASENAME%os.path.basename(wn_datarun)
                    )
        celltypes_dict = io.get_celltypes_dict(class_path)
        wn_cellids = sorted(wn_vcd.get_cell_ids())
        ns_cellids = sorted(ns_vcd.get_cell_ids())
        cellids_dict = pp.map_wn_to_ns_cellids(
                            ns_vcd,ns_cellids,wn_vcd,
                            wn_cellids,celltypes_dict,
                        )

        outdir = os.path.join('../tmp/',dataset,ns_datarun)

        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        fnameout = os.path.join(outdir,cfg.CELLIDS_DICT_FNAME)
        np.save(fnameout,cellids_dict,allow_pickle=True)

        # Bin the spike train.
        if "trigger" not in ns_datapath_dict[movie][dataset]:
            frames_per_ttl = cfg.FRAMES_PER_TTL
        else:
            trigger = ns_datapath_dict[movie][dataset]['trigger'] # frac FS
            frames_per_ttl = int(cfg.MONITOR_FS * trigger)

        frame_times = pp.get_frame_times(
                            ns_vcd,
                            frames_per_ttl,
                            frames_per_movie 
                      )
        binned_spikes = pp.get_binned_spikes(
                            ns_vcd,
                            cellids_dict['ns_cellids'],
                            frame_times
                        )

        # Reshape into a tensor of size movie,cells,time,repeat.
        n_movies,n_cells,frames_per_movie = binned_spikes.shape
        n_repeats = ns_datapath_dict[movie][dataset]['n_rep']
        
        if n_repeats == 1:
            binned_spikes = binned_spikes[...,None]
        else:
            n_unique_movies = ns_datapath_dict[movie][dataset]['n_sec']
            diff = n_unique_movies * n_repeats - n_movies
            binned_spikes = np.r_[
                              binned_spikes,
                              np.ones((diff,n_cells,frames_per_movie)) * np.nan
                            ]
            binned_spikes = np.moveaxis(
                                np.reshape(
                                    binned_spikes,
                                    (n_unique_movies,
                                    n_repeats,n_cells,
                                    frames_per_movie),
                                    order='F'
                                ),
                                1,-1
                            )

        fnameout = os.path.join(outdir,cfg.BINNED_SPIKES_FNAME)
        np.save(fnameout,binned_spikes)

        # Convert to firing rate by smoothing with boxcar.
        boxcar = pp.boxcar(cfg.BOXCAR_WINDOW)
        firing_rate = pp.smooth_spiketrains(
                            binned_spikes,
                            boxcar
                      )
        
        # Take the mean firing rate over trials and smooth with Gaussian.
        mean_firing_rate = np.nanmean(firing_rate,axis=-1,keepdims=True)
        gaussian = pp.gaussian(cfg.TAU,cfg.KERNEL_T)
        gaussian /= np.sum(gaussian)
        smoothed_firing_rate = pp.smooth_spiketrains(
                                        mean_firing_rate,
                                        gaussian
                                )
        fnameout = os.path.join(outdir,cfg.SMOOTHED_FIRING_RATE_FNAME)
        np.save(fnameout,smoothed_firing_rate)

        # Get a Gaussian kernel and smooth the spiketrain.
        gaussian = pp.gaussian(cfg.TAU,cfg.KERNEL_T)
        gaussian /= np.sum(gaussian)
        smoothed_spikes = pp.smooth_spiketrains(
                                    binned_spikes,
                                    gaussian,
                                    aggregated=True
                          )
        fnameout = os.path.join(outdir,cfg.SMOOTHED_SPIKES_FNAME)
        np.save(fnameout,smoothed_spikes)
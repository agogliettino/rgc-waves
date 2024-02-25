import numpy as np
from scipy import signal
import sys
sys.path.insert(0,'/home/agogliet/gogliettino/projects/nsem/repos/waves/')
import src.config as cfg
import pdb

""" 
Preprocessing to bin spike trains and get stimulus tensors and ultimately write
a training and testing partition to disk.
"""

def map_wn_to_ns_cellids(ns_vcd,ns_cellids,wn_vcd,
                         wn_cellids,celltypes_dict,
                         corr_dict = {'on parasol': .95,'off parasol': .95,
                                       'on midget': .95,'off midget': .95,
                                       'a1': .95},
                         mask=False,n_sigmas=None):
    """
    Maps WN to NS EIs according to a threshold value of correlation. Computes
    EI power over space, both with and without masking (user choice). Does a
    pass over the NS cellids and finds the corresponding WN cell. If none is
    found the cell doesn't get mapped (does not appear in the dictionary).

    Also writes a field for the normalized x,y locations for each RF.

    Parameters:
        ns_vcd: natural scenes vision data object
        ns_cellids: natural scenes cellids to map
        wn_cellids: white noise cellids to map
        celltypes_dict: dictionary mapping white noise cell ids to celltype.
    """

    channel_noise = wn_vcd.channel_noise

    # Initialize a dictionary and loop over the cells.
    cellids_dict = dict()

    for key in ['wn_to_ns','ns_to_wn','celltypes']:
        cellids_dict[key] = dict()

    for wn_cell in wn_cellids:

        # Get the cell type and write as well.
        celltype = celltypes_dict[wn_cell].lower()

        # Hardcode these for now TODO FIXME
        if "on" in celltype and "parasol" in celltype:
            celltype = 'on parasol'
        elif "off" in celltype and "parasol" in celltype:
            celltype = "off parasol"
        elif "on" in celltype and "midget" in celltype:
            celltype = 'on midget'
        elif "off" in celltype and "midget" in celltype:
            celltype = 'off midget'
        elif "a1" in celltype:
            celltype = 'a1'
        else:
            continue

        # If masking, only look at the significant indices. 
        wn_cell_ei = wn_vcd.get_ei_for_cell(wn_cell).ei

        if mask and n_sigmas is not None:
            sig_inds = np.argwhere(np.abs(np.amin(wn_cell_ei,axis=1))
                                   > n_sigmas * channel_noise).flatten()
            wn_cell_ei_power = np.zeros(wn_cell_ei.shape[0])
            wn_cell_ei_power[sig_inds] = np.sum(wn_cell_ei[sig_inds,:]**2,
                                                axis=1)
        else:
            wn_cell_ei_power = np.sum(wn_cell_ei**2,axis=1)

        corrs = []

        for ns_cell in ns_cellids:
            ns_cell_ei = ns_vcd.get_ei_for_cell(ns_cell).ei

            if mask and n_sigmas is not None:
                sig_inds = np.argwhere(np.abs(np.amin(ns_cell_ei,axis=1))
                                       > n_sigmas * channel_noise).flatten()
                ns_cell_ei_power = np.zeros(ns_cell_ei.shape[0])
                ns_cell_ei_power[sig_inds] = np.sum(ns_cell_ei[sig_inds,:]**2,
                                                axis=1)
            else:
                ns_cell_ei_power = np.sum(ns_cell_ei**2,axis=1)

            corr = np.corrcoef(wn_cell_ei_power,ns_cell_ei_power)[0,1]
            corrs.append(corr)

        # Take the cell with the largest correlation.
        if np.max(corrs) < corr_dict[celltype]:
            continue

        max_ind = np.argmax(np.asarray(corrs))
        cellids_dict['wn_to_ns'][wn_cell] = ns_cellids[max_ind]
        cellids_dict['ns_to_wn'][ns_cellids[max_ind]] = wn_cell
        cellids_dict['celltypes'][wn_cell] = celltype

        # Once the cell has been mapped, remove it (hack) FIXME.
        ns_cellids.remove(ns_cellids[max_ind]) 

    # Loop over the sorted NS cellids, get the normalized x,y locations.
    wn_field_x = int(wn_vcd.runtimemovie_params.width)
    wn_field_y = int(wn_vcd.runtimemovie_params.height)
    image_scalar = int(cfg.NS_FIELD_X / wn_field_x)
    cellids_dict['ns_cellids'] = sorted(cellids_dict['ns_to_wn'].keys())
    cellids_dict['rfs'] = []
    cellids_dict['rfs_for_vis'] = [] # with an offset for visualization

    for ns_cell in cellids_dict['ns_cellids']:
        wn_cell = cellids_dict['ns_to_wn'][ns_cell]
        sta_fit = wn_vcd.get_stafit_for_cell(wn_cell)
        mu_x = sta_fit.center_x * image_scalar
        mu_y = sta_fit.center_y * image_scalar
        cellids_dict['rfs_for_vis'].append(
                                    np.asarray(
                                        [mu_x,mu_y]
                                    ) 
                            )
        
        # Subtract off field height for computation-used x,y and get in microns 
        mu_y = (wn_field_y - sta_fit.center_y) * image_scalar
        cellids_dict['rfs'].append(
                                np.asarray(
                                    [mu_x,mu_y]
                                ) * cfg.STIXEL_SIZE * cfg.MICRONS_PER_PIXEL
                            )
    
    cellids_dict['rfs'] = np.asarray(cellids_dict['rfs'])
    cellids_dict['rfs_for_vis'] = np.asarray(cellids_dict['rfs_for_vis'])

    # Make a NS cell indices by celltype.
    cellids_dict['ns_cell_inds_by_celltype'] = dict()

    for celltype in cfg.CELLTYPES_OF_INT:
        cellids_dict['ns_cell_inds_by_celltype'][celltype] = []
    
    ns_to_wn = cellids_dict['ns_to_wn']
    
    for cc,ns_cell in enumerate(cellids_dict['ns_cellids']):
        celltype = cellids_dict['celltypes'][ns_to_wn[ns_cell]]
        cellids_dict['ns_cell_inds_by_celltype'][celltype].append(cc)

    for celltype in cfg.CELLTYPES_OF_INT:
        cellids_dict['ns_cell_inds_by_celltype'][celltype] =\
                        np.asarray(
                            cellids_dict['ns_cell_inds_by_celltype'][celltype]
                        )

    return cellids_dict

def get_frame_times(vcd,frames_per_ttl,frames_per_movie=None,unravel=False):
    """
    Gets times of each frame of the stimulus by linear interpolating the ttl
    times
    Parameters:
        vcd: vision data table object
        frames_per_ttl: the number of stimulus frames (set) to be between ttls
        frames_per_movie: if applicable, number of frames per movie
        unravel: boolean indicating to return a matrix or a vector
    Returns:
        vector of the approximate frame times, or a matrix in which the 
        second dim is the frames in each movie.
    """
    ttl_times = (vcd.ttl_times.astype(float) / cfg.FS) * 1000 # Convert to ms.
    frame_times = []

    for i in range(ttl_times.shape[0]-1):
        frame_times.append(np.linspace(ttl_times[i],
                               ttl_times[i+1],
                               frames_per_ttl,
                               endpoint=False))

    frame_times = np.asarray(frame_times).ravel()

    # If not unravel, just return the vector.
    if not unravel and frames_per_movie is None:
        return frame_times

    # Remove the change so we have round number.
    n_frames = frame_times.shape[0]

    while n_frames % frames_per_movie !=0:
        n_frames -= 1

    frame_times = frame_times[0:n_frames]

    # Reshape into a matrix.
    n_movies = int(n_frames / frames_per_movie)
    frame_times = np.reshape(frame_times,(n_movies,frames_per_movie))

    return frame_times

def get_binned_spikes(vcd,cells,frame_times):
    """
    Bins spiketrains with 1 ms bins.
    Parameters:
        vcd: initialized vision cell data table object from vision loader with 
             include_neurons=True
        cells: a list of cells for which binned spiketrains are wanted.
        frame_times: either a vector or matrix of frame times.
    Returns:
        binned spiketrain matrix of size n_ms x cells.
    """

    # Bin the spike train using the frame times as bin edges.
    binned_spikes = []
    movie_duration = int((frame_times.shape[1] / cfg.MONITOR_FS) * 1000)
    bin_edges = np.linspace(0,movie_duration,movie_duration+1)

    for i in range(frame_times.shape[0]):
        movie_tmp = []
        t0 = frame_times[i,0]
        t_end = frame_times[i,-1] + (1 / cfg.MONITOR_FS * 1000)

        for cc,cell in enumerate(cells):
            spike_times = (vcd.get_spike_times_for_cell(cell) 
                            / cfg.FS) * 1000 # ms.
            spike_times = spike_times[
                            np.where(
                                (spike_times > t0) & 
                                (spike_times < t_end)
                            )
                          ]
            movie_tmp.append(np.histogram(spike_times - t0,bin_edges)[0])

        movie_tmp = np.asarray(movie_tmp)
        binned_spikes.append(movie_tmp)
    
    binned_spikes = np.asarray(binned_spikes)

    # If the frame times was a matrix, make this into a tensor.
    '''
    if len(frame_times.shape) == 1:
        binned_spikes = np.swapaxes(binned_spikes,0,1)
    else:
        n_movies,frames_per_movie = frame_times.shape
        n_cells = binned_spikes.shape[0]
        binned_spikes = np.reshape(binned_spikes,
                            (n_cells,n_movies,frames_per_movie)
                        )
        binned_spikes = np.moveaxis(binned_spikes,1,0)
    '''
    #return binned_spikes.astype(np.uint8)
    return binned_spikes

def gaussian(tau,t):
    """ Gaussian kernel """
    return 1 / np.sqrt(2 * np.pi) * np.exp(-t**2 / tau**2)

def boxcar(window):
    """Moving average filter"""
    return np.ones(window) / window

def smooth_spiketrains(binned_spikes_tensor,kernel,aggregated=False):
    """
    Smooths the spiketrain by convolving with a kernel.
    Can either smooth individual or aggregated over trials. Expects as an 
    argument a tensor of size movie x cell x time x repeats.
    """

     # If aggregated, sum over the last dimension.
    smoothed_spiketrains_tensor = []
    
    if aggregated:
        binned_spikes_tensor = np.nansum(binned_spikes_tensor,axis=-1)
        
        for mm in range(binned_spikes_tensor.shape[0]):
            tmp = []
            for cc in range(binned_spikes_tensor.shape[1]):
                tmp.append(np.convolve(kernel,
                                       binned_spikes_tensor[mm,cc,...],
                                       mode='same'))
            smoothed_spiketrains_tensor.append(np.asarray(tmp))
    else:
        for mm in range(binned_spikes_tensor.shape[0]):
            movie_tmp = []
            for cc in range(binned_spikes_tensor.shape[1]):
                cell_tmp = []
                for tt in range(binned_spikes_tensor.shape[3]):
                    cell_tmp.append(np.convolve(kernel,
                                                binned_spikes_tensor[mm,cc,:,tt],
                                                mode='same'))
                movie_tmp.append(np.swapaxes(np.asarray(cell_tmp),0,1))
            smoothed_spiketrains_tensor.append(np.asarray(movie_tmp))
        
    return np.asarray(smoothed_spiketrains_tensor)
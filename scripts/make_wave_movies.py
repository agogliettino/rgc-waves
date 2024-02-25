"""
Script to make NS wave movies for each data set.
"""
import sys
import os
import numpy as np
sys.path.insert(0,'/home/agogliet/gogliettino/projects/nsem/repos/waves/')
import src.config as cfg
import src.visualize as vi
import src.io_util as io
from scipy import signal
from IPython.core.display import display, HTML
import pdb

ns_datapath_dict = np.load(
                    '../tmp/ns_datapath_dict.npy',
                    allow_pickle=True
                   ).item()

for movie in ns_datapath_dict:

    for dataset in ns_datapath_dict[movie]:
            
        ns_datarun = ns_datapath_dict[movie][dataset]['ns_datarun']
        dirin = os.path.join('../tmp/',dataset,ns_datarun)
        fnamein = os.path.join(dirin,cfg.CELLIDS_DICT_FNAME)
        cellids_dict = np.load(fnamein,allow_pickle=True).item()
        fnamein = os.path.join(dirin,cfg.SMOOTHED_FIRING_RATE_FNAME)
        smoothed_firing_rate = np.load(fnamein).squeeze()
        n_movies = ns_datapath_dict[movie][dataset]['n_sec']
        frames_per_movie = ns_datapath_dict[movie][dataset]['frames_per_movie']

        if "nsbrownian" in movie.lower():
            movie_path = os.path.join(
                            cfg.PARENT_NS_MOVIE['NSbrownian'],
                            movie
                        )
        else:
            assert False

        ns_tensor = io.get_raw_movie(
                        movie_path,
                        n_movies * frames_per_movie
                    )[...,0] # since grayscale.
                
        n_frames,n_pixels_y,n_pixels_x = ns_tensor.shape
        ns_tensor = np.moveaxis(
                        np.reshape(
                            ns_tensor,
                            (n_movies,
                            frames_per_movie,
                            n_pixels_y,n_pixels_x)
                        ),
                        1,-1
                    )
        
        for i in range(smoothed_firing_rate.shape[0]):

            print(i)

            smoothed_spikes_resample = np.maximum(
                                        signal.resample(
                                            smoothed_firing_rate[i,...],
                                            frames_per_movie,
                                            axis=-1),0
                                        )
            
            anim = vi.plot_wave(
                        cellids_dict,
                        smoothed_spikes_resample,
                        ns_tensor[i,...]
                    )
            
            HTML(anim.to_jshtml())
            anim.save(f'../tmp/wave_{i}.gif',writer='pillow', fps=cfg.FPS) 


        

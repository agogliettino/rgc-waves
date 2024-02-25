import numpy as np
import os
import sys
import config as cfg
import rawmovie as rm
import pdb

"""
Miscellaneous io utilities.
"""

def get_inds_from_block_design(n_blocks,n_train,n_test,train_first=True):
    ind = 0
    train_ind =  0
    train_inds = []

    while ind < (n_train + n_test) * n_blocks:
        train_inds.append(ind)

        if (train_ind + 1) % n_train == 0:
            ind += n_test+1
            train_ind = 0
        else:
            ind +=1
            train_ind +=1

    return np.asarray(train_inds)

def get_celltypes_dict(class_path,lower=True):
    """
    Gets the celltype dictionary mapping IDs to celltypes from
    a text file.
    Parameters:
        class_path: full path to text file of cell types
        lower: boolean indicating whether to lowercase the strings
    Returns:
        dictionary mapping IDs to celltype.
    """

    f = open(class_path)
    celltypes_dict = dict()

    for j in f:
        tmp = ""

        for jj,substr in enumerate(j.split()[1:]):
            tmp +=substr

            if jj < len(j.split()[1:])-1:
                tmp += " "

        if lower:
            tmp = tmp.lower()

        celltypes_dict[int(j.split()[0])] = tmp

    f.close()

    return celltypes_dict

def get_raw_movie(raw_movie_path,n_frames):
    """
    Utility for getting natural scenes raw movies from disk 
    Parameters:
        raw_movie_path: path to stimulus
        n_frames: number of unique frames (int)
    Returns: 
        stimulus tensor of size n,y,x,c 
    """
    assert os.path.isfile(raw_movie_path), "Stimulus provided not found."

   # Initialize the stimulus object.
    rm_obj = rm.RawMovieReader(raw_movie_path)
    stimulus_tensor,_ = rm_obj.get_frame_sequence(0,n_frames)

    return stimulus_tensor
"""
Constants used throughout the project.
"""
import numpy as np
import sys
sys.path.insert(0,'/home/agogliet/gogliettino/projects/nsem/repos/waves/')

PARENT_ANALYSIS = '/Volumes/Analysis/'
PARENT_NS_MOVIE = dict()
PARENT_NS_MOVIE['NSbrownian'] =\
        '/Volumes/Data/Stimuli/movies/eye-movement/'\
        'current_movies/NSbrownian/NSbrownian_3000_movies/'
DEVICES = ['cuda:0','cuda:1','cuda:2','cuda:3','cpu']

PARENT_NS_MOVIE['INbrownian'] = '/Volumes/Data/Stimuli/movies/imagenet/INbrownian/'
PARENT_NS_MOVIE['NSinterval'] = '/Volumes/Data/Stimuli/movies/eye-movement/current_movies/NSinterval/'
PARENT_NS_MOVIE['ImageNet'] = '/Volumes/Data/Stimuli/movies/imagenet'
CLASSIFICATION_BASENAME = "%s.classification_agogliet.txt"
FRAMES_PER_TTL = 100
FS = 20000 # Hz
MS_PER_S = 1000
MS_OF_INTEREST = 250 # wave duration
MONITOR_FS = 120
TAU = 10 # ms, for Gaussian kernel
CELLTYPES_OF_INT = [
        'on parasol','off parasol','a1','on midget','off midget',
        ]
BOXCAR_WINDOW = 25
N_THREADS = 36
KERNEL_T = np.linspace(-50,50,101) # in ms for the kernel.
CELLIDS_DICT_FNAME = 'cellids_dict.npy'
BINNED_SPIKES_FNAME = 'binned_spikes.npy'
SMOOTHED_SPIKES_FNAME = 'smoothed_spikes.npy'
SMOOTHED_FIRING_RATE_FNAME = 'smoothed_firing_rate.npy'
MOTION_ENERGY_FNAME = 'motion_energy_dict.npy'
VEC_SUM_FNAME = 'vector_sum_dict.npy'
SIG_IND_DICT_FNAME = 'sig_ind_dict.npy'
META_FNAME = 'meta_dict.npy'
META_STIMULUS_FNAME = 'meta_stimulus.npy'

# Stuff for bootstrapping.
N_SIM = 10000 # simulations
ALPHA = .01 # significance level
SEED = 1995

# Stuff for plotting.
SIZE_SCALAR = 1000
NS_FIELD_Y = 160
NS_FIELD_X = 320
PLAYBACK_INTERVAL = 83.3
COLORS = {
        'on parasol': 'C0','off parasol': 'C3','a1': 'C4',
        'on midget': 'C2','off midget': 'C1'
        }
TITLES = {
        'on parasol': 'ON parasol','off parasol': 'OFF parasol','a1': 'A1',
        'on midget': 'ON midget','off midget': 'OFF midget'
        }

STIXEL_SIZE = 2 # for most of the NS movies.
MICRONS_PER_PIXEL = 5.5 # For CRT with 6.5x
N_FRAMES = 30 # Show 30 frames at 120 Hz.
EI_SIZE_SCALAR = 2.5
FPS = 15 # frames per second for gif writing.

# For motion energy calculation.
THETAS = np.linspace(0,360,25) # degrees
SPEEDS = np.linspace(1,30,30) # in µm/ms

# Define bounds for cardinal directions
"""
DIRECTIONS = {
    'left': lambda theta: np.where((theta > 135) & (theta < 225))[0],
    'right': lambda theta: np.where(
                                (theta < 45) & (theta > 0) | 
                                (theta > 315)
                           )[0],   
    'up': lambda theta: np.where((theta > 45) & (theta < 135))[0],
    'down': lambda theta: np.where((theta > 225) & (theta < 315))[0]
}
"""
DIRECTIONS = {
    'left': lambda theta: np.where((theta > 157.5) & (theta < 225))[0],
    'right': lambda theta: np.where(
                                (theta < 22.5) & (theta > 0) | 
                                (theta > 337.5)
                           )[0],   
    'up': lambda theta: np.where((theta > 67.5) & (theta < 112.5))[0],
    'down': lambda theta: np.where((theta > 247.5) & (theta < 292.5))[0]
}
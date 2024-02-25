import sys
import numpy as np
import os
import seq_nmf as snf
import pdb
import importlib as il
import torch
import time
from progressbar import *

dataset_name = '2016-04-21-1'
wn_datarun = 'kilosort_data006/data006'
mb_datarun = 'kilosort_data009/data009'
wn_datapath = os.path.join(cfg.PARENT_ANALYSIS,
                          dataset_name,wn_datarun)
mb_datapath = os.path.join(cfg.PARENT_ANALYSIS,
                          dataset_name,mb_datarun)

indir = os.path.join('../tmp/',dataset_name,mb_datarun)
fnamein = os.path.join(indir,cfg.CELLIDS_DICT_FNAME)
cellids_dict = np.load(fnamein,allow_pickle=True).item()

fnamein = os.path.join(indir,cfg.SMOOTHED_SPIKES_FNAME)
smoothed_spikes = np.load(fnamein)

# Run seq-NMF
celltype_inds = cellids_dict['ns_cell_inds_by_celltype']
celltype = 'on parasol'
X_ = smoothed_spikes[celltype_inds[f'{celltype}'],:].copy()
pdb.set_trace()
lamdas = np.logspace(-1,-6,10)
lamdas = np.append(lamdas,0)
MAX_ITER = 100
N,_ = X_.shape
torch.cuda.set_device(cfg.DEVICE)
SEED = 1995
np.random.seed(SEED)
#K = 12 # exactly 12 differnet factors here in the moving bar stimulus.
#K = 20 # exactly 12 differnet factors here in the moving bar stimulus.
#Ks = [6,12,24]
Ks = [12]
L = 1500
lamda_L1_W = 0
lamda_L1_H = 0
lamda_ortho_H = 0
lamda_ortho_W = 0
shift = True

smooth_kernel = np.ones((1,(2*L)-1))
#inv_eye = (~(np.eye(K).astype(bool))).astype(float)
pad = np.zeros((N,L))
X = np.c_[pad,X_,pad]
_,T = X.shape
small_number = np.max(X)*1e-6

for K in Ks:
    inv_eye = (~(np.eye(K).astype(bool))).astype(float)
    
    for ll,lamda in enumerate(lamdas):
        
        #if ll not in [lamdas.shape[0] - 1]: continue
        lamda = 0
        t0_master = time.time()
        data_dict = {'hyperparams': 
                    {'lamda': None, 'K': None,
                        'L': None, 'MAX_ITER': None},
                    'data': {'X': None,'X_hat': None,
                            'W': None,'H': None,
                            'cost': None,'x_ortho_cost': None
                            }
                }
        W_init = np.max(X)*np.random.rand(N,K,L)
        H_init = np.max(X)*np.random.rand(K,T) / (np.sqrt(T/3))
        pdb.set_trace()
        X_hat = snf.reconstruct(W_init,H_init)
        cost = np.zeros((MAX_ITER+1, 1))
        X_hat = snf.reconstruct(W_init,H_init)
        cost[0] = np.sqrt(np.mean((X - X_hat)**2))
        W = W_init.copy()
        H = H_init.copy()
        widgets = ['Running interative optimization: ',
                Percentage(), ' ', Bar(marker='*',
            left='[',right=']'),' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=MAX_ITER)
        pbar.start()

        for i in range(MAX_ITER):
            
            if i == MAX_ITER-1:
                lamda = 0

            WTX,WTX_hat = snf.compute_cnmf_H_update_terms(X,X_hat,W)
            dRdH = snf.compute_dRdH(WTX,H,smooth_kernel,inv_eye,
                                    lamda,lamda_L1_H,lamda_ortho_H
                )

            # Update H.
            H = H * WTX / (WTX_hat + dRdH + np.finfo(H.dtype).eps)

            if shift:
                W,H = snf.shift_factors(W,H)
                W += small_number

            W,H = snf.normalize_W_H(W,H)
            W = snf.update_W(W,H,X,smooth_kernel,inv_eye,
                            lamda,lamda_L1_W,lamda_ortho_W
                )
            X_hat = snf.reconstruct(W,H)
            cost[i+1] = np.sqrt(np.mean((X - X_hat)**2))
            pbar.update(i)
        
        pbar.finish()
        t_end = time.time()
        print(f'done. took {t_end - t0_master} seconds.')
        t0 = t_end 
        print('computing percent power of loadings ... ')
        loadings = snf.compute_loading_pcnt_pwr(X,W,H)
        t_end = time.time()
        print(f'done. took {t_end - t0} seconds')
        t0 = t_end
        inds = np.argsort(loadings)[::-1]
        W = W[:,inds,:]
        H = H[inds,:]
        print('computing x_ortho cost ... ')
        x_ortho_cost = snf.compute_x_ortho_cost(X,W,H,smooth_kernel,inv_eye)
        t_end = time.time()
        print(f'done. took {t_end - t0} seconds.')
        data_dict['hyperparams']['lamda'] = lamdas[ll]
        data_dict['hyperparams']['MAX_ITER'] = MAX_ITER
        data_dict['hyperparams']['L'] = L
        data_dict['hyperparams']['K'] = K
        data_dict['data']['X'] = X[:,L:-L]
        data_dict['data']['X_hat'] = X_hat[:,L:-L]
        data_dict['data']['W'] = W
        data_dict['data']['H'] = H[:,L:-L]
        data_dict['data']['cost'] = cost[-1]
        data_dict['data']['cost_all_iter'] = cost
        data_dict['data']['x_ortho_cost'] = x_ortho_cost
        
        #dirout =  f'../tmp/{dataset_name}/{mb_datarun}/seq-NMF-testing/'
        dirout =  f'../tmp/{dataset_name}/{mb_datarun}/seq-NMF-testing/matlab-test/'
        
        if not os.path.isdir(dirout):
            os.makedirs(dirout)
        
        ctype = celltype.replace(' ','')
        fnameout = os.path.join(
                        dirout,f'snf_out_{ctype}_li{ll}_L{L}_K{K}_I{MAX_ITER}.npy'
                    )
        #np.save(fnameout,data_dict,allow_pickle=True)
        
        t_end_master = time.time()
        print(f'total seq-NMF {t_end_master - t0_master} seconds.')
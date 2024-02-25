import sys
import os
sys.path.insert(0,'/home/agogliet/gogliettino/projects/nsem/repos/waves/')
import src.config as cfg
cfg.N_THREADS = 4

os.environ["OMP_NUM_THREADS"] = f"{cfg.N_THREADS}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{cfg.N_THREADS}" 
os.environ["MKL_NUM_THREADS"] = f"{cfg.N_THREADS}" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{cfg.N_THREADS}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{cfg.N_THREADS}" 

import torch
torch.set_num_threads(cfg.N_THREADS)

import numpy as np
from numpy import linalg
import torch.nn.functional as F
import time
from scipy import signal,stats
import pdb

def reconstruct(W,H):
    N,K,L = W.shape
    _,T = H.shape
    pad = np.zeros((K,L))
    H = np.c_[pad,H,pad]
    T = T+2*L
    X_hat = np.zeros((N,T))
    W = torch.tensor(W).to(torch.float32).to(cfg.DEVICE)
    H = torch.tensor(H).to(torch.float32).to(cfg.DEVICE)
    X_hat = torch.tensor(X_hat).to(torch.float32).to(cfg.DEVICE)
    
    with torch.no_grad():
        for tau in np.arange(0,L):
            X_hat += torch.matmul(W[...,tau],
                                  torch.roll(H,tau,dims=1)
                            )
    
    W = W.detach().cpu().numpy()
    H = H.detach().cpu().numpy()
    X_hat = X_hat.detach().cpu().numpy()
    
    with torch.cuda.device(cfg.DEVICE):
        torch.cuda.empty_cache()
    
    return X_hat[:,L:-L]

def shift_factors(W,H):
    N,K,L = W.shape
    _,T = H.shape
    
    if L == 1:
        return W,H
    
    center = np.maximum(np.floor(L / 2),1)
    W_pad = np.c_[np.zeros((N,K,L)),W,np.zeros((N,K,L))]
    
    for k in range(K):
        tmp = np.sum(W[:,k,:],axis=0)
        cmass = np.maximum(
                    np.floor(
                        np.sum(
                            tmp * np.arange(1,len(tmp) + 1)
                        ) / np.sum(tmp)
                    ),
                    1
               )
        
        # Kludge TODO:
        if np.isnan(cmass):
            cmass = 1
            
        W_pad[:,k,:] = np.roll(W_pad[:,k,:],int(center - cmass),axis=1)
        H[k,:] = np.roll(H[k,:],int(cmass - center))
    
    return W_pad[...,L:-L],H

def compute_cnmf_H_update_terms(X,X_hat,W):
    _,K,L = W.shape
    _,T = X.shape
    WTX = torch.tensor(
                np.zeros((K,T))
          ).to(torch.float32).to(cfg.DEVICE)
    WTX_hat = torch.tensor(
                    np.zeros((K,T))
              ).to(torch.float32).to(cfg.DEVICE)
    X = torch.tensor(X).to(torch.float32).to(cfg.DEVICE)
    X_hat = torch.tensor(X_hat).to(torch.float32).to(cfg.DEVICE)
    W = torch.tensor(W).to(torch.float32).to(cfg.DEVICE)
    
    with torch.no_grad():
        for l in range(L):
            X_shifted = torch.roll(X,-l,dims=1)
            X_hat_shifted = torch.roll(X_hat,-l,dims=1)
            WTX += torch.matmul(
                        torch.t(W[:, :, l]),
                        X_shifted
                   )
            WTX_hat += torch.matmul(
                            torch.t(W[:, :, l]),
                            X_hat_shifted
                        )
    
    WTX = WTX.detach().cpu().numpy()
    WTX_hat = WTX_hat.detach().cpu().numpy()
    X = X.detach().cpu().numpy()
    X_hat = X_hat.detach().cpu().numpy()
    W = W.detach().cpu().numpy()
    
    with torch.cuda.device(cfg.DEVICE):
        torch.cuda.empty_cache()
    
    return WTX,WTX_hat

def compute_dRdH(WTX,H,smooth_kernel,inv_eye,lamda,lamda_L1_H,lamda_ortho_H):
    WTX = torch.tensor(WTX).to(torch.float32).to(cfg.DEVICE)
    H = torch.tensor(H).to(torch.float32).to(cfg.DEVICE)
    smooth_kernel = torch.tensor(smooth_kernel).to(torch.float32).to(cfg.DEVICE)
    inv_eye = torch.tensor(inv_eye).to(torch.float32).to(cfg.DEVICE)
    
    if lamda == 0:
        dRdH = 0
    else:
        with torch.no_grad():
            conv_out = F.conv2d(
                        WTX[None,None,...],
                        smooth_kernel[None,None,...],
                        padding='same'
                       ).squeeze()
            mat_out = torch.matmul(inv_eye,conv_out).detach().cpu().numpy()
            WTX = WTX.detach().cpu().numpy()
            conv_out = conv_out.detach().cpu().numpy()
            dRdH = lamda * mat_out
        
    if lamda_ortho_H == 0:
        dHHdH = 0
    else:
        with torch.no_grad():
            conv_out = F.conv2d(
                            H[None,None,...],
                            smooth_kernel[None,None,...],
                            padding='same'
                       ).squeeze()
            mat_out = torch.matmul(inv_eye,conv_out).detach().cpu().numpy()
            smooth_kernel = smooth_kernel.detach().cpu().numpy()
            inv_eye = inv_eye.detach().cpu().numpy()
            dHHdH = lamda_ortho_H * mat_out
        
    dRdH += (lamda_L1_H + dHHdH)
    
    with torch.cuda.device(cfg.DEVICE):
        torch.cuda.empty_cache()
    
    return dRdH

def normalize_W_H(W,H):
    N,K,L = W.shape
    norms = np.sqrt(np.sum(H**2,axis=1)).T
    H = np.matmul(np.diag(1 / (norms +  np.finfo(H.dtype).eps)),H)
    
    for l in range(L):
        W[...,l] = np.matmul(W[...,l],np.diag(norms))
    
    return W,H

def update_W(W,H,X,smooth_kernel,inv_eye,lamda,lamda_L1_W,lamda_ortho_W):
    N,K,L = W.shape
    X_hat = reconstruct(W,H)
    X = torch.tensor(X).to(torch.float32).to(cfg.DEVICE)
    smooth_kernel = torch.tensor(smooth_kernel).to(torch.float32).to(cfg.DEVICE)
    
    with torch.no_grad():
        X_S = F.conv2d(
                    X[None,None,...],
                    smooth_kernel[None,None,...],
                    padding='same'
              )
        
    smooth_kernel = smooth_kernel.detach().cpu().numpy()
    H = torch.tensor(H).to(torch.float32).to(cfg.DEVICE)
    X_hat = torch.tensor(X_hat).to(torch.float32).to(cfg.DEVICE)
    W = torch.tensor(W).to(torch.float32).to(cfg.DEVICE)
    inv_eye = torch.tensor(inv_eye).to(torch.float32).to(cfg.DEVICE)
    
    if lamda_ortho_W > 0:
        with torch.no_grad():
            W_flat = torch.sum(W,dims=2)
            dWWdW = lamda_ortho_W * torch.matmul(W_flat,inv_eye)
            W_flat = W_flat.detach().cpu().numpy()
    else:
        dWWdW = 0
    
    with torch.no_grad():
        for l in range(L):
            H_shifted = torch.roll(H,l,dims=1)
            X_HT = torch.matmul(X,torch.t(H_shifted))
            X_hat_HT = torch.matmul(X_hat,torch.t(H_shifted))
            
            if lamda == 0:
                dRdW = torch.tensor([0]).to(torch.float32).to(cfg.DEVICE)
            else:
                dRdW = lamda * torch.matmul(
                                    torch.matmul(
                                        X_S,torch.t(H_shifted)
                                    ),
                                    inv_eye
                               )
            """
            if lamda > 0:
                dRdW = lamda * torch.matmul(
                    torch.matmul(
                        X_S,torch.t(H_shifted)
                    ),
                    inv_eye
                )
            else:
                dRdW = 0
                
            """ 
            dRdW += (lamda_L1_W + dWWdW)
            W[...,l] = W[...,l] * X_HT / (X_hat_HT + dRdW + 
                                          torch.finfo(X_hat_HT.dtype).eps)
    
    X = X.detach().cpu().numpy()
    X_S = X_S.detach().cpu().numpy()
    X_HT = X_HT.detach().cpu().numpy()
    X_hat_HT = X_hat_HT.detach().cpu().numpy()

    if type(dRdW) != int:
        dRdW = dRdW.detach().cpu().numpy()

    W = W.detach().cpu().numpy()
    X_hat = X_hat.detach().cpu().numpy()
    inv_eye = inv_eye.detach().cpu().numpy()
    
    with torch.cuda.device(cfg.DEVICE):
        torch.cuda.empty_cache()
    
    return W

def compute_loading_pcnt_pwr(V,W,H):
    loadings = [] 
    K = H.shape[0]
    var_v = np.sum(V**2)
    
    for i in range(K):
        W_H = reconstruct(W[:,i,:][:,None,:],H[i,:][None,:])
        loadings.append(np.sum(2 * V * W_H - W_H**2)/var_v)
    
    loadings = np.asarray(loadings)
    loadings[np.where(loadings < 0)] = 0
    
    return loadings

def compute_x_ortho_cost(X,W,H,smooth_kernel,inv_eye):
    N,K,L = W.shape
    _,T = X.shape
    WTX = torch.tensor(np.zeros((K, T))).to(torch.float32).to(cfg.DEVICE)
    W = torch.tensor(W).to(torch.float32).to(cfg.DEVICE)
    inv_eye = torch.tensor(inv_eye).to(torch.float32).to(cfg.DEVICE)
    smooth_kernel = torch.tensor(smooth_kernel).to(torch.float32).to(cfg.DEVICE)
    H = torch.tensor(H).to(torch.float32).to(cfg.DEVICE)
    
    with torch.no_grad(): 
        for l in range(L):
            X_shifted = torch.tensor(
                            np.c_[X[:,l:T],np.zeros((N,l))]
                        ).to(torch.float32).to(cfg.DEVICE)
            WTX += torch.matmul(torch.t(W[...,l]),X_shifted)

        WTXSHT = torch.matmul(
                        F.conv2d(
                            WTX[None,None,...],
                            smooth_kernel[None,None,...],
                            padding='same'
                        ),
                        torch.t(H)
                ).squeeze()
        WTXSHT *= inv_eye
        
    WTX = WTX.detach().cpu().numpy()
    W = W.detach().cpu().numpy()
    H = H.detach().cpu().numpy()
    inv_eye = inv_eye.detach().cpu().numpy()
    WTXSHT = WTXSHT.detach().cpu().numpy()
    X_shifted = X_shifted.detach().cpu().numpy()
    smooth_kernel = smooth_kernel.detach().cpu().numpy()
    
    with torch.cuda.device(cfg.DEVICE):
        torch.cuda.empty_cache()
        
    return linalg.norm(WTXSHT,ord=1)

def get_normalized_cost(xortho_costs,costs,window_size=3):
    b = (1/window_size)*np.ones((1,window_size)).squeeze()
    a = 1
    rs = signal.filtfilt(b,a,xortho_costs);

    min_rs = np.percentile(xortho_costs,0)
    max_rs = np.percentile(xortho_costs,100)

    rs = (rs-min_rs)/(max_rs-min_rs);
    r = (xortho_costs-min_rs)/(max_rs-min_rs);

    cs = signal.filtfilt(b,a,costs);
    min_cs = np.percentile(costs,0)
    max_cs = np.percentile(costs,100);
    cs = (cs - min_cs)/(max_cs-min_cs);
    c = (costs - min_cs)/(max_cs-min_cs);
    
    #return r,c
    return rs,cs

def test_factor_significance(X_test,W,p,n_null=None):
    mask = np.sum(np.sum(W > 0,axis=0,keepdims=True),axis=2) == 0
    W_flat = np.sum(W,axis=2)
    mask |= (np.max(W_flat,axis=0)**2 > .999 * np.sum(W_flat**2,axis=0))
    mask = ~(mask.astype(bool)).squeeze()
    W = W[:,mask,:]
    N,K,L = W.shape
    _,T = X_test.shape
    X_test = torch.tensor(X_test).to(torch.float32).to(cfg.DEVICE)
    W = torch.tensor(W).to(torch.float32).to(cfg.DEVICE)

    if n_null is None:
        n_null = int(np.ceil(K/p) * 2)
        
    skew_null = np.zeros((K,n_null))

    for n in range(n_null):
        W_null = torch.zeros((N,K,L)).to(torch.float32).to(cfg.DEVICE)

        for k in range(K):
            for ni in range(N):
                W_null[ni,k,:] = torch.roll(
                                        W[ni,k,:],
                                        np.random.randint(L),
                                )
        
        WTX = torch.zeros((K,T)).to(torch.float32).to(cfg.DEVICE)
        
        with torch.no_grad():
            for l in range(L):
                X_shifted = torch.roll(X_test,-l+2,dims=1)
                WTX += torch.matmul(torch.t(W_null[...,l]),X_shifted)
        
        WTX = WTX.detach().cpu().numpy() 
        skew_null[:,n] = stats.skew(WTX,bias=True,axis=1)
    
    WTX = torch.zeros((K,T)).to(torch.float32).to(cfg.DEVICE)
    
    with torch.no_grad(): 
        for l in range(L):
            X_shifted = torch.roll(X_test,-l+2,dims=1)
            WTX += torch.matmul(torch.t(W[...,l]),X_shifted)
    
    WTX = WTX.detach().cpu().numpy() 
    skew = stats.skew(WTX,bias=True,axis=1)
    pvals = [] 
    
    for k in range(K):
         pvals.append((1 + np.sum(skew_null[k,:] > skew[k])) / n_null)
        
    pvals = np.asarray(pvals)
    all_pvals = np.zeros(mask.shape)
    all_pvals[~mask] = np.inf
    all_pvals[mask] = pvals
    pvals = all_pvals
    
    with torch.cuda.device(cfg.DEVICE):
        torch.cuda.empty_cache()
        
    return pvals, pvals <= p / K
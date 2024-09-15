import torch as th
import numpy as np

from .eigh import *

def upper(X):
    '''
    Upper operation for vectorising tangent space matrices. 
    Adopted from https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/tangentspace.py

    Args:
        X (torch.Tensor): Input SPD matrices of size (*,n,n)

    Returns:
        torch.Tensor: Output P, which is the vectorised upper triangular parts of X of size (*,n*(n+1)/2)
    '''
    n = X.shape[-1]
    if X.shape[-2] != n:
        raise ValueError("Matrices must be square")
    idx = th.triu_indices(n,n,device=X.device)
    coeffs = (np.sqrt(2) * th.triu(th.ones((n, n)), 1) + th.eye(n)).to(X.device, dtype=X.dtype)[idx[0], idx[1]]
    T = coeffs * X[..., idx[0], idx[1]]
    return T


##############################################
### Log-Euclidean operations
##############################################

def tensor_log(t):
    u, s, v = th.svd(t) 
    return u @ th.diag_embed(th.log(s)) @ v.transpose(-2,-1)

def dist_log_euc(A, B):
    '''
    Compute the log Euclidean distance between SPD matrices in A and B.

    If A and B have the same shape, compute pair-wise distances. 
    If one of A/B has shape (n,n), compute the distances to this matrix 
    from all matrices in the other input (B/A).

    Args:
        A (torch.Tensor): batch of SPD matrices with shape (*,n,n)
        B (torch.Tensor): batch of SPD matrices with shape (*,n,n)

    Returns:
        torch.Tensor: distance scalars of shape (*,)
    '''
    inner_term = tensor_log(A) - tensor_log(B)
    inner_multi = inner_term @ inner_term.transpose(-2,-1)
    _, s, _= th.svd(inner_multi)
    return th.sqrt(th.sum(s, dim=-1))

def log_eucl_mean(X):
    '''
    Compute the Log Euclidean mean of X. 

    Args:
        X (torch.Tensor): batch of SPD matrices of shape (*,n,n)
    
    Returns:
        torch.Tensor: the Log Euclidean mean of X - a single SPD matrix of shape (n,n)
    '''
    X = LogEig.apply(X)
    output = th.mean(X, dim=-3)
    return ExpEig.apply(output)

def weighted_log_eucl_mean(W, X):
    '''
    Compute the weighted Log Euclidean mean of X. 

    Args:
        W (torch.Tensor): the weight vector of shape (N,) 
        X (torch.Tensor): batch of SPD matrices of shape (*, n, n)

    Returns:
        torch.Tensor: the weighted Log Euclidean mean of X
    '''
    n = X.shape[-1]

    X1 = LogEig.apply(X).flatten(-2)
    output = W @ X1 #(*,-1)
    shape = list(output.shape[:-1])
    shape.extend([n,n]) # (*,n,n)
    output = output.view(*shape)
    output = ExpEig.apply(output)
    return output

#########################
### Adopted from SPDNetBN
#########################

def dist_riemann(x,y):
    '''
    Riemannian distance between SPD matrices x and SPD matrix y

    Args:
        x (torch.Tensor): batch of SPD matrices of shape (*,n,n)
        y (torch.Tensor): single SPD matrix of shape (n,n)

    Returns:
        torch.Tensor: distance scalers of shape (*,)
    '''
    return LogEig.apply(CongrG(x,y,'neg')).flatten(-2).norm(p=2,dim=-1)

def geodesic(A,B,t):
    '''
    Geodesic from A to B at step t

    Args:
        A (torch.Tensor): SPD matrix (n,n) to start from
        B (torch.Tensor): SPD matrix (n,n) to end at
        t (float): scalar parameter of the geodesic (not constrained to [0,1])

    Returns:
        torch.Tensor: SPD matrix (n,n) along the geodesic at step t
    '''
    C = CongrG(PowerEig.apply(CongrG(B,A,'neg'),t),A,'pos')
    return C

def bimap(X,W):
    '''
    Bilinear mapping function from SPDNet

    Args:
        X (torch.Tensor): Input matrix of shape (batch_size,n_in,n_in)
        W (torch.Tensor): Stiefel parameter of shape (n_in,n_out)

    Returns:
        torch.Tensor: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    return W.transpose(-2,-1) @ X @ W

def bimap_channels(X,W):
    '''
    Bilinear mapping function over multiple input and output channels

    Args:
        X (torch.Tensor): Input matrix of shape (batch_size,channels_in,n_in,n_in)
        W (torch.Tensor): Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)

    Returns:
        torch.Tensor: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    batch_size,_,_,_=X.shape
    channels_out,_,_,n_out=W.shape
    P=th.zeros(batch_size,channels_out,n_out,n_out,dtype=X.dtype,device=X.device)
    for co in range(channels_out):
        P[:,co,:,:]=th.sum(bimap(X, W[co]), dim=1)
    return P

def bimap_groups(X,W):
    '''
    Bilinear mapping function over multiple channels within group

    Args:
        X (torch.Tensor): Input matrix of shape (batch_size,groups,channels_in,n_in,n_in)
        W (torch.Tensor): Stiefel parameter of shape (groups,channels_in,n_in,n_out)
    
    Returns:
        torch.Tensor: Bilinearly mapped matrix of shape (batch_size,groups,n_out,n_out)
    '''
    batch_size,n_groups,_,_,_=X.shape
    _,_,_,n_out=W.shape
    P=th.zeros(batch_size,n_groups,n_out,n_out,dtype=X.dtype,device=X.device)
    for go in range(n_groups):
        P[:,go,:,:]=th.sum(bimap(X[:,go], W[go]), dim=1)
    return P

def LogG(x,X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    return CongrG(LogEig.apply(CongrG(x,X,'neg')),X,'pos')

def ExpG(x,X):
    """ Exponential mapping of x on the SPD manifold at X """
    return CongrG(ExpEig.apply(CongrG(x,X,'neg')),X,'pos')

def karcher_step(x,G,alpha):
    '''
    One step in the Karcher flow

    Args:
        x (torch.Tensor): batch of SPD matrices of shape (*,n,n)
        G (torch.Tensor): SPD matrix (n,n) to update
        alpha (float): step size

    Returns:
        torch.Tensor: updated SPD matrix (n,n)
    '''
    x_log=LogG(x,G)
    G_tan=x_log.mean(dim=0).squeeze()
    G=ExpG(alpha*G_tan,G)
    return G

def BaryGeom(x, k=40, alpha=1):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow

    Args:
        x (torch.Tensor): batch of SPD matrices of shape (*,n,n) to compute the barycenter of
        k (int): number of iterations
        alpha (float): step size
    
    Returns:
        torch.Tensor: Riemannian mean of the input SPD matrices
    '''
    with th.no_grad():
        G=th.mean(x,dim=0).squeeze()
        for _ in range(k):
            G=karcher_step(x,G,alpha)
        return G

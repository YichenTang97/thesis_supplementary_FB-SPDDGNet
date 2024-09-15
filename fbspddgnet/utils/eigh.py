"""
Eigenvalue operations.

Following functions are adopted from SPDNet implemented in the supplemental material of paper 
# "Riemannian batch normalization for SPD neural networks" 
# (https://proceedings.neurips.cc/paper/2019/hash/6e69ebbfad976d4637bb4b39de261bf7-Abstract.html)
"""


import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Function as F

def repeat_last_dimension(S):
    pattern = [1] * len(S.shape) + [S.shape[-1]]
    return S[...,None].repeat(*pattern)

def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class

    Args:
        P (torch.Tensor): SPD matrices of size (*, n, n)
        op (object): Operator object used for eigenvalue modification
        eig_mode (str, optional): Eigenvalue computation mode. Defaults to 'svd'.
        param (object, optional): Additional parameters for the operator. Defaults to None.

    Returns:
        torch.Tensor: Modified symmetric matrices of size (*, n, n)
        torch.Tensor: Eigenvectors of P
        torch.Tensor: Eigenvalues of P
        torch.Tensor: Modified eigenvalues of P
    '''
    if(eig_mode=='eig'):
        S,U = th.linalg.eig(P)
        U,S = U.real, S.real
    elif(eig_mode=='svd'):
        U,S,_ = th.svd(P)
    S_fn = op.fn(S,param)
    X = U @ th.diag_embed(S_fn) @ U.transpose(-2,-1)
    return X,U,S,S_fn

def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input:
        dx: The gradient of the output with respect to the modified symmetric matrices. Shape: (*, n, n)
        U: The eigenvectors of the input symmetric matrices. Shape: (*, n, n)
        S: The eigenvalues of the input symmetric matrices. Shape: (*, n)
        S_fn: The modified eigenvalues of the input symmetric matrices. Shape: (*, n)
        op: The non-linear eigenvalue modification operator.
        param: Additional parameters for the operator (optional).
    Output:
        dp: The gradient of the output with respect to the input symmetric matrices. Shape: (*, n, n)
    '''
    S_fn_deriv = th.diag_embed(op.fn_deriv(S, param))
    SS = repeat_last_dimension(S)
    SS_fn = repeat_last_dimension(S_fn)
    L = (SS_fn - SS_fn.transpose(-2, -1)) / (SS - SS.transpose(-2, -1))
    L[L == -np.inf] = 0
    L[L == np.inf] = 0
    L[th.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(-2, -1) @ dx @ U)
    dp = dp.to(th.float)
    U = U.to(th.float)
    dp = U @ dp @ U.transpose(-2, -1)
    return dp

def CongrG(P, G, mode):
    """
    Congruence transformation of SPD matrices.

    Args:
        P: SPD matrices of size (*, n, n)
        G: Tensor of shape (n, n) representing the matrix to perform the congruence by.
        mode: String indicating the mode of congruence. Can be 'pos' or 'neg'.

    Returns:
        Tensor of shape (*, n, n) representing the congruence by sqm(G) or sqminv(G).

    Raises:
        ValueError: If the mode is not 'pos' or 'neg'.
    """
    if mode == 'pos':
        GG = SqmEig.apply(G)
    elif mode == 'neg':
        GG = SqminvEig.apply(G)
    else:
        raise ValueError("Invalid mode. Mode must be 'pos' or 'neg'.")

    PP = GG @ P @ GG
    return PP


class Log_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.log(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 1/S

class LogEig(F):
    """
    Logarithm eigenvalue computation for SPD matrices.

    Args:
        P (torch.Tensor): Input SPD matrices of size (*,n,n)

    Returns:
        torch.Tensor: Output X, which is the log eigenvalues matrices of size (*,n,n)
    """
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Log_op)


class Re_op():
    """ Rectify function and its derivative """
    _threshold=1e-4
    @classmethod
    def fn(cls,S,param=None):
        return nn.Threshold(cls._threshold,cls._threshold)(S)
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (S>cls._threshold).double()

class ReEig(F):
    """
    Rectified eigenvalue computation for SPD matrices.

    Args:
        P (torch.Tensor): Input SPD matrices of size (*,n,n)

    Returns:
        torch.Tensor: Rectified eigenvalues matrices of size (*,n,n)
    """
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)


class Exp_op():
    """ Exp function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.exp(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return th.exp(S)

class ExpEig(F):
    """
    Exponential eigenvalue computation for SPD matrices. 

    Args:
        P (torch.Tensor): Input SPD matrices of size (*,n,n)

    Returns:
        torch.Tensor: Exponential eigenvalues matrices of size (*,n,n)
    """
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Exp_op, eig_mode='eig')
        ctx.save_for_backward(U, S, S_fn)
        return X
    
    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Exp_op)


class Sqm_op():
    """ Square root function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 0.5/th.sqrt(S)

class SqmEig(F):
    """
    Square root eigenvalue computation for SPD matrices. P^(1/2).

    Args:
        P (torch.Tensor): Input SPD matrices of size (*, n, n).

    Returns:
        torch.Tensor: Square root eigenvalues matrices of size (*, n, n).
    """
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqm_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqm_op)


class Sqminv_op():
    """ Inverse square root function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return 1/th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return -0.5/th.sqrt(S)**3

class SqminvEig(F):
    """
    Inverse square root eigenvalue computation for SPD matrices. P^(-1/2).

    Args:
        P (torch.Tensor): Input SPD matrices of size (*,n,n)

    Returns:
        torch.Tensor: Output inverse square root eigenvalues matrices of size (*,n,n)
    """
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqminv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqminv_op)


class Power_op():
    """ Power function and its derivative """
    _power=1
    @classmethod
    def fn(cls,S,param=None):
        return S**cls._power
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (cls._power)*S**(cls._power-1)

class PowerEig(F):
    """
    Power eigenvalue computation for SPD matrices. P^{power}.

    Args:
        P (tensor): Input SPD matrices of size (*,n,n).
        power (float): The power value for eigenvalue computation.

    Returns:
        tensor: Output X, power eigenvalues matrices of size (*,n,n).
    """
    @staticmethod
    def forward(ctx, P, power):
        Power_op._power = power
        X, U, S, S_fn = modeig_forward(P, Power_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Power_op), None


class Inv_op():
    """ Inverse function and its derivative """
    @classmethod
    def fn(cls,S,param=None):
        return 1/S
    @classmethod
    def fn_deriv(cls,S,param=None):
        return th.log(S)

class InvEig(F):
    """
    Inverse eigenvalue computation for SPD matrices. 

    Args:
        P (tensor): Input SPD matrices of size (*,n,n).

    Returns:
        tensor: Output X, inverse eigenvalues matrices of size (*,n,n).
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Inv_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Inv_op)


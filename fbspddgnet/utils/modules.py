import geoopt
import torch
import numpy as np
import torch.nn as nn

from einops import rearrange

from . import functional as fn
from . import eigh as eigh

dtype = torch.float32

class DomainAdaptiveBatchNorm(nn.Module):
    """
    Riemannian geometry-based domain adaptation module. 
    It performs centring, scaling, rotating (optional) and parallel shifting (bias, optional) steps on SPD matrices.

    Params:
        n (int): Size of SPD matrices.
        momentum (float, optional): Momentum for updating running mean and variance. Default is 0.1.
        eps (float, optional): A value added to variance for numerical stability. Default is 1e-05.
        rotate (bool, optional): Whether to perform rotation step. Default is True.
        bias (bool, optional): Whether to perform parallel shifting (bias) step. Default is True.
        R (torch.Tensor, optional): ManifoldParameter representing the rotation matrix. If None, an identity matrix of size n will be used. Default is None.
        B (torch.Tensor, optional): ManifoldParameter representing the parallel shifting matrix. If None, an identity matrix of size n will be used. Default is None.
    """
    def __init__(self, n, momentum=0.1, eps=1e-05, rotate=True, bias=True, R=None, B=None):
        super(__class__, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.rotate = rotate
        self.bias = bias
        self.running_mean = torch.eye(n, dtype=dtype)
        self.running_var = torch.tensor(1., dtype=dtype)
        if rotate:
            self.R = geoopt.ManifoldParameter(torch.eye(n, dtype=dtype), 
                                                manifold=geoopt.manifolds.CanonicalStiefel()) if R is None else R
        if bias:
            self.B = geoopt.ManifoldParameter(torch.eye(n, dtype=dtype),
                                                manifold=geoopt.manifolds.SymmetricPositiveDefinite()) if B is None else B

    def set_mean_var(self, mean, var):
        """
        Set the running mean and variance.

        Params:
            mean (torch.Tensor): The running mean.
            var (torch.Tensor): The running variance.
        """
        with torch.no_grad():
            self.running_mean.data = mean
            self.running_var.data = var
    
    def set_R(self, R):
        """
        Set the rotation matrix.

        Params:
            R (torch.Tensor): The rotation matrix.
        """
        with torch.no_grad():
            self.R.data = R
    
    def set_bias(self, B):
        """
        Set the parallel shifting matrix.

        Params:
            B (torch.Tensor): The parallel shifting matrix.
        """
        with torch.no_grad():
            self.B.data = B
    
    def forward(self, X):
        """
        Forward pass of the DomainAdaptiveBatchNorm module.

        Params:
            X (torch.Tensor): Input SPD matrices.

        Returns:
            torch.Tensor: Output SPD matrices after centring, scaling, rotating (optional), and parallel shifting (optional) steps.
        """
        N, q, n, n = X.shape
        X_batched = rearrange(X, "N q n1 n2 -> (N q) n1 n2")

        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_var = self.running_var.to(X.device)
        
        if self.training:
            mean = fn.BaryGeom(X_batched) # use Karcher flow
            var = torch.mean(fn.dist_riemann(X_batched, mean).squeeze() ** 2)
            with torch.no_grad():
                self.running_mean.data = fn.geodesic(self.running_mean, mean, self.momentum)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            p = torch.sqrt(1. / (var + self.eps))
            X_batched = fn.CongrG(X_batched, mean, 'neg')
            X_batched = eigh.PowerEig.apply(X_batched, p)
        else:
            p = torch.sqrt(1. / (self.running_var + self.eps))
            X_batched = fn.CongrG(X_batched, self.running_mean, 'neg')
            X_batched = eigh.PowerEig.apply(X_batched, p)
        if self.rotate:
            X_batched = fn.bimap(X_batched, self.R)
        if self.bias:
            X_batched = fn.CongrG(X_batched, self.B, 'pos')
        return rearrange(X_batched, '(N q) n1 n2 -> N q n1 n2', N=N)

class DomainGeneralisationBN(nn.Module):
    def __init__(self, domains, n, momentum=0.1, eps=1e-05, rotate=True, bias=True, parallel=True, R=None, B=None):
        """
        Initializes the DomainGeneralisationBN module.

        Args:
            domains (list): List of domain names.
            n (int): Size of SPD matrices.
            momentum (float, optional): Momentum for the batch normalization. Defaults to 0.1.
            eps (float, optional): Small value added to the variance to avoid division by zero. Defaults to 1e-05.
            rotate (bool, optional): Whether to apply rotation to the SPD matrices. Defaults to True.
            bias (bool, optional): Whether to include bias term in the batch normalization. Defaults to True.
            parallel (bool, optional): Whether to use parallel computation on GPU for different domains. Defaults to True.
            R (torch.Tensor, optional): ManifoldParameter representing the rotation matrix. If None, an identity matrix of size n will be used. Default is None.
            B (torch.Tensor, optional): ManifoldParameter representing the parallel shifting matrix. If None, an identity matrix of size n will be used. Default is None.

        See also:
            DomainAdaptiveBatchNorm
        """
        super(__class__, self).__init__()
        # encode domains as integers
        self.d_idx = np.arange(len(domains))
        self.domains = domains
        self.bns = nn.ModuleList([
            DomainAdaptiveBatchNorm(n, momentum, eps, rotate, bias, R, B) for _ in domains
        ])
        self.parallel = parallel

    def get_d_idx(self, d):
        """
        Get the index of a domain.

        Args:
            d (str): Domain name.

        Returns:
            int: Index of the domain.
        """
        return self.domains.index(d)

    def set_mean_var(self, mean, var, d):
        """
        Set the mean and variance for a specific domain.

        Args:
            mean (torch.Tensor): Geometric mean SPD matrix.
            var (torch.Tensor): SPD matrices variance.
            d (str): Domain name.
        """
        self.bns[self.get_d_idx(d)].set_mean_var(mean, var)
    
    def set_R(self, R, d):
        """
        Set the rotation matrix for a specific domain.

        Args:
            R (torch.Tensor): Rotation matrix.
            d (str): Domain name.
        """
        self.bns[self.get_d_idx(d)].set_R(R)

    def set_bias(self, bias, d):
        """
        Set the bias SPD matrix for a specific domain.

        Args:
            bias (torch.Tensor): Bias SPD matrix.
            d (str): Domain name.
        """
        self.bns[self.get_d_idx(d)].set_bias(bias)

    def forward(self, X, ds):
        """
        Forward pass of the module.

        Args:
            X (torch.Tensor): Input SPD matrices of shape (bs, q, n_ch, n_ch).
            ds (list): List of domains.

        Returns:
            torch.Tensor: Output tensor after applying batch normalization.
        """
        if X.is_cuda and self.parallel:
            cuda = torch.device('cuda')
            streams = [torch.cuda.Stream() for _ in range(len(self.domains))]
            # wait for all previous steps done
            for s in streams:
                s.wait_stream(torch.cuda.default_stream(cuda))

        # X_copy = torch.empty_like(X)
        for i, d in zip(self.d_idx, self.domains):
            if d in ds:
                s = ds == d
                Xs = X[s]
                if X.is_cuda and self.parallel:
                    with torch.cuda.stream(streams[i]):
                        X[s] = self.bns[i](Xs)
                else:
                    X[s] = self.bns[i](Xs)

        # Wait for all streams to finish
        if X.is_cuda and self.parallel:
            torch.cuda.synchronize()

        return X
    

class Upper(nn.Module):
    """
    Upper operation for vectorising tangent space matrices. 
    Adopted from https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/tangentspace.py

    Args:
        P (torch.Tensor): Input SPD matrices of size (*,n,n)

    Returns:
        torch.Tensor: Output P, which is the vectorised upper triangular parts of X of size (*,n*(n+1)/2)
    """
    def forward(self, P):
        return fn.upper(P)

#####################################################
### BiMap, LogEig, ReEig layers adopted from SPDNetBN 
#####################################################

class BiMap(nn.Module):
    """
    Bilinear Mapping Module for SPD Matrices.

    This module takes as input a batch of SPD matrices and performs bilinear mapping on them.
    The input matrix X has shape (batch_size, hi) where hi is the size of the input SPD matrices (ni, ni).
    The output matrix P has shape (batch_size, ho) where ho is the size of the output bilinearly mapped matrices (no, no).
    The Stiefel parameter has shape (ho, hi, ni, no).

    Args:
        hi (int): Size of the input SPD matrices.
        ho (int): Size of the output bilinearly mapped matrices.
        ni (int): Size of the input SPD matrices (ni, ni).
        no (int): Size of the output bilinearly mapped matrices (no, no).
    """
    def __init__(self, hi, ho, ni, no):
        super(BiMap, self).__init__()
        self._hi = hi
        self._ho = ho
        self._ni = ni
        self._no = no
        self.increase_dim = None
        if no > ni:
            self.increase_dim = SPDIncreaseDim(ni, no)
            self._ni = no
        self._W = geoopt.ManifoldParameter(torch.empty(ho, hi, self._ni, self._no, dtype=dtype), 
                                           manifold=geoopt.manifolds.CanonicalStiefel())
        nn.init.orthogonal_(self._W)

    def forward(self, X):
        """
        Forward pass of the BiMap module.

        Args:
            X (torch.Tensor): Input batch of SPD matrices with shape (batch_size, hi).

        Returns:
            torch.Tensor: Output batch of bilinearly mapped matrices with shape (batch_size, ho).
        """
        if self.increase_dim:
            X = self.increase_dim(X)
        return fn.bimap_channels(X, self._W)
    
class BiMapGroups(nn.Module):
    """
    A module that performs bilinear mapping within group of SPD matrices.

    Args:
        groups (int): The number of groups (e.g., frequency bands).
        segments (int): The number of segments (e.g., temporal segments).
        ni (int): The input matrix size.
        no (int): The output matrix size.

    Note:
        The input X should have contain (batch_size, groups, segments) SPD matrices of size (ni, ni).
        The output P will have (batch_size, groups) bilinearly mapped matrices of size (no, no).
        The Stiefel parameter has shape (groups, segments, ni, no).
    """
    def __init__(self, groups, segments, ni, no):
        super(BiMapGroups, self).__init__()
        self._groups = groups
        self._segments = segments
        self._ni = ni
        self._no = no
        self.increase_dim = None
        if no > ni:
            self.increase_dim = SPDIncreaseDim(ni, no)
            self._ni = no
        self._W = geoopt.ManifoldParameter(torch.empty(groups, segments, self._ni, self._no, dtype=dtype),
                                           manifold=geoopt.manifolds.CanonicalStiefel())
        nn.init.orthogonal_(self._W)

    def forward(self, X):
        """
        Performs the forward pass of the module.

        Args:
            X (torch.Tensor): The input tensor of shape (batch_size, groups, segments, ni, ni).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, groups, no, no).
        """
        if self.increase_dim:
            X = self.increase_dim(X)
        return fn.bimap_groups(X, self._W)


class LogEig(nn.Module):
    """
    Logarithm eigenvalue computation for SPD matrices.

    Args:
        P (torch.Tensor): Input SPD matrices of size (*,n,n)

    Returns:
        torch.Tensor: Output X, which is the log eigenvalues matrices of size (*,n,n)
    """
    def forward(self,P):
        return fn.LogEig.apply(P)

class ReEig(nn.Module):
    """
    Rectified eigenvalue computation for SPD matrices.

    Args:
        P (torch.Tensor): Input SPD matrices of size (*,n,n)

    Returns:
        torch.Tensor: Rectified eigenvalues matrices of size (*,n,n)
    """
    def forward(self,P):
        return fn.ReEig.apply(P)
    
########################################################
### E2R operation adopted from MAtt 
### https://github.com/CECNL/MAtt
########################################################

class signal2spd(nn.Module):
    """
    Operation for computing sample covariance matrices from EEG signals.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the signal2spd module.

        Args:
            x (torch.Tensor): Input EEG signals of shape (*, n_ch, n_t)

        Returns:
            torch.Tensor: Covariance matrices of shape (*, n_ch, n_ch)
        """
        # Save the original shape and reshape to 3D
        original_shape = x.shape
        x = x.view(-1, original_shape[-2], original_shape[-1])

        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean

        cov = x@x.permute(0, 2, 1)
        cov = cov/(x.shape[-1]-1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra

        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=cov.device).repeat(x.shape[0], 1, 1)
        cov = cov+(1e-5*identity)

        # Reshape back
        cov = cov.view(*original_shape[:-1], original_shape[-2])
        
        return cov

class E2R(nn.Module):
    """
    Module for converting Euclidean EEG signals into Riemannian covariance matrices

    Args:
        segments (int): Number of temporal segments to split the signals into.
        overlap (float): Overlapping ratio between segments.
    """
    def __init__(self, segments, overlap=0):
        super().__init__()
        self.segments = segments
        self.overlap = overlap
        self.signal2spd = signal2spd()
    
    def forward(self, x):
        """
        Forward pass of the E2R module.

        Params:
            x: Input EEG signals of shape (N, *, n_ch, n_t)

        Returns:
            x: Covariance matrices of shape (N, q, *, n_ch, n_ch)
        """
        # let T = time series length, n = number of segments, m = overlap ratio, e = epoch length, then we get:
        # n*e - (n-1)*m*e = T, so we can compute e as:
        ep_len = x.shape[-1] / ((1-self.overlap) * self.segments + self.overlap)
        ep_len = int(np.ceil(ep_len))
        overlap = int(ep_len*self.overlap)

        x_list = []
        for n in range(self.segments):
            start = n*ep_len - n*overlap
            if n+1 != self.segments:
                x_list.append(self.signal2spd(x[...,start:start+ep_len]))
            else:
                x_list.append(self.signal2spd(x[...,start:]))
        x = torch.stack(x_list).transpose(0, 1)
        return x # (N, q, *, n_ch, n_ch)

class SPDIncreaseDim(nn.Module):
    """
    Module for increasing the dimension of SPD matrices.

    Args:
        input_size (int): Size of the input SPD matrices.
        output_size (int): Size of the output SPD matrices.
    """
    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.register_buffer('eye', torch.eye(output_size, input_size))
        add = torch.as_tensor([0] * input_size + [1] * (output_size-input_size), dtype=dtype)
        self.register_buffer('add', torch.diag(add))

    def forward(self, input):
        """
        Forward pass of the SPDIncreaseDim module.

        Args:
            input (torch.Tensor): Input SPD matrices of size (*,n,n)

        Returns:
            torch.Tensor: Output SPD matrices of size (*,m,m)
        """
        output = self.add + self.eye @ input @ self.eye.t()
        return output

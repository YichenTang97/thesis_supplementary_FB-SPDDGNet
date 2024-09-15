import torch
import itertools
import torch.nn as nn

from einops.layers.torch import Rearrange

from sklearn.base import TransformerMixin, BaseEstimator
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space

from .utils.modules import *


###########################################################################
### Proposed models (FB-SPDBN, FB-SPDDGBN)
### These models require Conv2dWithConstraint defined in this file to work
###########################################################################

class FB_SPDBN(nn.Module):
    def __init__(self, args):
        """
        Initializes the FB_SPDBN model.

        Args:
            args: An object containing the model configuration parameters.

        Parameter in args:
            n_classes: Number of classes.
            n_ch: Number of EEG channels.
            n_bands: Number of frequency bands.
            n_segments: Number of temporal segments for computing covariance matrices from -- feature extraction (FE) block.
            overlap: Overlap ratio between segments.
            conv_ch_1: Number of spatial channels to output from the first 2-D convolution layer of FE block.
            conv_ch_2: Number of spatial channels to output from the second 2-D convolution layer of FE block.
            conv_t: Temporal convolution kernel size in the second 2-D convolution layer of FE block.
            bi_ho_1: Number of output channels in the first BiMap layer.
            bi_no_1: Number of output SPD matrix size in the first BiMap layer.
            bi_ho_2: Number of output channels in the second BiMap layer.
            bi_no_2: Number of output SPD matrix size in the second BiMap layer.
            norm_momentum: Batch normalization momentum.
        """
        super(FB_SPDBN, self).__init__()

        # load args
        self.n_classes = args.n_classes
        self.n_ch = args.n_ch
        self.norm_momentum = args.norm_momentum
        self.n_bands = args.n_bands
        self.n_segments = args.n_segments
        self.overlap = args.overlap

        self.conv_ch_1 = args.conv_ch_1
        self.conv_ch_2 = args.conv_ch_2
        self.conv_t = args.conv_t
        self.bi_ho_1 = args.bi_ho_1
        self.bi_no_1 = args.bi_no_1
        self.bi_ho_2 = args.bi_ho_2
        self.bi_no_2 = args.bi_no_2

        # expecting input to be (n_ep, n_band, n_ch, n_t)
        self.proc = nn.Sequential(
            # spatial filter
            Conv2dWithConstraint(self.n_bands, self.n_bands*self.conv_ch_1, (self.n_ch, 1), groups=self.n_bands), # perform group (band) wise convolution
            nn.BatchNorm2d(self.n_bands*self.conv_ch_1),
            # spatial-temporal filter
            Conv2dWithConstraint(self.n_bands*self.conv_ch_1, self.n_bands*self.conv_ch_2, (1, self.conv_t), padding=(0, int(self.conv_t/2)), groups=self.n_bands), # perform group (band) wise convolution
            nn.BatchNorm2d(self.n_bands*self.conv_ch_2),
            Rearrange('e (b c) 1 t -> e b c t', b=self.n_bands),
            E2R(segments=self.n_segments, overlap=self.overlap), # (bs, q, b, c, c)
            Rearrange('e q b c1 c2 -> e (b q) c1 c2')
        )

        self.bimap = BiMap(self.n_bands*self.n_segments, self.bi_ho_1, self.conv_ch_2, self.bi_no_1)
        self.bn = DomainAdaptiveBatchNorm(self.bi_no_1, self.norm_momentum, rotate=False)
        self.latent = nn.Sequential(
            ReEig(), # (bs, b, c, c),
            BiMap(self.bi_ho_1, self.bi_ho_2, self.bi_no_1, self.bi_no_2),
            ReEig(),
            LogEig(),
            Upper(),
            nn.Flatten(),
        )

        n = self.bi_no_2
        m = self.n_classes
        q = self.bi_ho_2
        
        self.out = nn.Sequential(
            nn.Linear(int(q*(n*(n+1)/2)), m, bias=True),
            nn.Softmax(dim=-1)
        )

        self.init_params()
    
    def init_params(self):
        """
        Initializes the parameters of the model.
        """
        for module in self.proc.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias"):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.out[0].weight, gain=1)
        self.out[0].bias.data.fill_(0)

    def freeze(self):
        """
        Freezes the parameters of the model.
        """
        for module in [self.proc, self.bimap, self.bn, self.latent]:
            for param in module.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        """
        Unfreezes the parameters of the model.
        """
        for module in [self.proc, self.bimap, self.bn, self.latent]:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, X, return_latent=False):
        '''
        Forward pass of the FB_SPDBN model.

        Args:
            X: Input tensor of shape (bs, band, n_ch, n_t).
            return_latent: Boolean flag indicating whether to return the latent representation.

        Returns:
            If return_latent is True, returns a tuple (pred, out) where pred is the predicted output and out is the latent representation.
            If return_latent is False, returns the predicted output.
        '''
        # ensure 4D input
        if len(X.shape) == 3 and self.n_bands == 1:
            X = X.unsqueeze(1)

        out = self.proc(X) # (bs, [q, b], c, c)
        out = self.bimap(out)
        out = self.bn(out)
        out = self.latent(out)
        pred = self.out(out)

        if return_latent:
            return pred, out
        else:
            return pred
    

class FB_SPDDGBN(FB_SPDBN):
    def __init__(self, args, source_domains, target_domains, rotate=True, bias=True, parallel=True):
        """
        Initialize the FB_SPDDGBN model.

        Args:
            args: An object containing the model configuration parameters.
            source_domains: List of source domains.
            target_domains: List of target domains.
            rotate (bool, optional): Flag indicating whether to apply rotation in the domain-specific SPD domain generalisation (DG) layer. Defaults to True.
            bias (bool, optional): Flag indicating whether to apply bias (parallel shifting) in the domain-specific SPD DG layer. Defaults to True.
            parallel (bool, optional): Flag indicating whether to use parallel computation in CUDA for different domains in DG layer. Defaults to True.

        See also:
            FB_SPDBN
        """
        super().__init__(args)

        self.rotate = rotate
        self.bias = bias
        self.parallel = parallel
        self.source_domains = source_domains
        self.target_domains = target_domains

        self.sbn = DomainGeneralisationBN(self.source_domains, self.bi_no_1, 
                                            momentum=self.norm_momentum, rotate=self.rotate, bias=self.bias, parallel=self.parallel)
        self.sbn_target = DomainGeneralisationBN(self.target_domains, self.bi_no_1, 
                                            momentum=self.norm_momentum, rotate=self.rotate, bias=self.bias, parallel=self.parallel)

        self.r_weights = nn.Parameter(torch.ones(self.bi_ho_1, dtype=dtype)) # q weights for weighted mean of q spd matrices in each trial
        self.prototypes = torch.eye(self.bi_no_1,dtype=dtype).reshape((1,self.bi_no_1,self.bi_no_1))
        self.prototypes = self.prototypes.repeat(self.n_classes,1,1) # (n_classes, n, n)
        self.prototypes = geoopt.ManifoldParameter(self.prototypes, 
                                                   manifold=geoopt.manifolds.SymmetricPositiveDefinite())

    def freeze(self):
        """
        Freeze the parameters of the model.
        """
        for module in [self.proc, self.bimap, self.sbn, self.latent]:
            for param in module.parameters():
                param.requires_grad = False
        self.prototypes.requires_grad = False
        self.r_weights.requires_grad = False
    
    def unfreeze(self):
        """
        Unfreeze the parameters of the model.
        """
        for module in [self.proc, self.bimap, self.sbn, self.latent]:
            for param in module.parameters():
                param.requires_grad = True
        self.prototypes.requires_grad = True
        self.r_weights.requires_grad = True
    
    def domain_adapt(self, X, ds, k=40):
        '''
        Perform domain adaptation on the input data.

        Args:
            X (torch.Tensor): Input data of shape (bs, p, n, n), where bs is the batch size,
                              p is the number of segments, and n is the size of each spd matrix.
            ds (torch.Tensor): Domain labels for trials.
            k (int, optional): Number of steps for Karcher flow. Defaults to 40.
        '''
        # ensure 4D input
        if len(X.shape) == 3 and self.n_bands == 1:
            X = X.unsqueeze(1)

        with torch.no_grad():
            X = self.bimap(self.proc(X)) # (bs, [q, b], c, c)
            R=torch.mean(torch.stack([sn.R.data for sn in self.sbn.bns]), dim=0)
            B=torch.stack([sn.B.data for sn in self.sbn.bns]) # (n_domains, n, n)
            B=fn.BaryGeom(B, k=k)

            for d in self.target_domains:
                X_d = X[ds==d]
                X_batched = rearrange(X_d, "N q n1 n2 -> (N q) n1 n2")
                mean = fn.BaryGeom(X_batched, k=k)
                var  = torch.mean(fn.dist_riemann(X_batched, mean).squeeze() ** 2)
                self.sbn_target.set_mean_var(mean, var, d)
                self.sbn_target.set_R(R, d)
                self.sbn_target.set_bias(B, d)

    def prototype_similarities(self):
        """
        Compute the pair-wise similarity between prototypes for different classes (using negative log transformed riemannian distances).

        Returns:
            torch.Tensor: Pair-wise similarities between prototypes.
        """
        similarities = []
        for i, j in itertools.combinations(range(self.n_classes), 2):
            similarities.append(-torch.log1p(fn.dist_riemann(self.prototypes[i], self.prototypes[j])))
        return torch.stack(similarities)
    
    def dist(self, X, y):
        """
        Compute the Riemannian distance between input spd matrices and class prototypes.

        Args:
            X (torch.Tensor): SPD matrices of shape (bs, p, n, n), where bs is the batch size,
                              p is the number of segments, and n is the size of each spd matrix.
            y (torch.Tensor): Target labels of shape (bs,) containing the class indices for each sample.

        Returns:
            torch.Tensor: Riemannian distances between input data and class prototypes.
        """
        bs, p, _, _ = X.shape
        dist = torch.zeros(bs, device=X.device)
        if self.prototypes.device != X.device:
            self.prototypes.to(X.device)
        
        X_weighted = fn.weighted_log_eucl_mean(self.r_weights, X) # (bs, n, n)
        for i in range(self.n_classes):
            if not i in y:
                continue
            s = y==i
            dist[s] = fn.dist_riemann(X_weighted[s], self.prototypes[i])
        return torch.log1p(dist)

    def forward(self, X, ds, y=None, on_source=True):
        '''
        Forward pass of the model.

        Args:
            X (torch.Tensor): Input EEG signals of shape (bs, band, n_ch, n_t).
            ds (torch.Tensor): Domain labels for trials.
            y (torch.Tensor, optional): Class labels. Defaults to None.
            on_source (bool, optional): Flag indicating whether the process is training on the source. Defaults to True.

        Returns:
            tuple: A tuple containing the predicted output and the distance to prototypes.
        '''
        # ensure 4D input
        if len(X.shape) == 3 and self.n_bands == 1:
            X = X.unsqueeze(1)

        out = self.proc(X) # (bs, [q, b], c, c)
        out = self.bimap(out)

        if on_source:
            out = self.sbn(out, ds)
            dist = self.dist(out, y) if not y is None else None
            sims = self.prototype_similarities()
        else:
            out = self.sbn_target(out, ds)
            dist = self.dist(out, y) if not y is None else None
            sims = None

        out = self.latent(out)
        pred = self.out(out)

        return pred, dist, sims


#############################
### Baseline models
#############################

class EEGNet(nn.Module):
    def __init__(self, args):
        '''
        Adopted from 
        https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
        and
        https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py
        '''
        super(EEGNet, self).__init__()

        self.n_classes = args.n_classes
        self.n_chs = args.n_chs
        self.n_t = args.n_t
        self.kernLength = args.kernLength
        self.F1 = args.F1
        self.D = args.D
        self.F2 = args.F2
        self.dropoutRate = args.dropoutRate
        self.dropoutType = args.dropoutType
        
        self.block1 = nn.Sequential(
            Rearrange('e c t -> e 1 c t'),
            # conv temporal
            nn.Conv2d(1, self.F1, (1, self.kernLength), padding=(0, self.kernLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            # conv spatial
            Conv2dWithConstraint(self.F1, self.F1*self.D, (self.n_chs, 1), max_norm=1, stride=1, groups=self.F1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F1*self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(self.dropoutRate) if self.dropoutType == 'Dropout' else nn.Dropout2d(self.dropoutRate)
        )
        
        self.block2 = nn.Sequential(
            # conv_separable_depth
            nn.Conv2d(self.F1*self.D, self.F1*self.D, (1, 16), stride=1, groups=self.F1*self.D, padding=(0, 16 // 2), bias=False),
            # conv_separable_point
            nn.Conv2d(self.F1*self.D, self.F2, (1, 1), stride=1, padding=(0, 0), bias=False),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(self.dropoutRate) if self.dropoutType == 'Dropout' else nn.Dropout2d(self.dropoutRate),
            nn.Flatten()
        )

        with torch.inference_mode():
            mock_x = torch.zeros(2, self.n_chs, self.n_t)
            latent_len = self.block2(self.block1(mock_x)).shape[1]
        
        self.out = nn.Sequential(
            nn.Linear(latent_len, self.n_classes),
            nn.Softmax(dim=1)
        )

        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias"):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def freeze(self):
        for module in [self.block1, self.block2]:
            for param in module.parameters():
                param.requires_grad = False
        
    def unfreeze(self):
        for module in [self.block1, self.block2]:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x, return_latent=False):
        '''
        x: tensor of shape (n_bs, n_ch, n_t)
        '''
        x = self.block1(x)
        x = self.block2(x)
        pred = self.out(x)

        if return_latent:
            return pred, x
        else:
            return pred


class Tensor_CSPNet(nn.Module):
    '''
    Tensor-CSPNet adopted from https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet/blob/main/utils/model.py
    '''
    def __init__(self, args, mlp=False):
        super(Tensor_CSPNet, self).__init__()

        self._mlp             = mlp
        self.n_bands          = args.n_bands
        self.n_segments       = args.n_segments
        self.channel_in       = args.n_bands*args.n_segments
 
        classes           = args.n_classes
        self.dims         = args.dims
        self.kernel_size  = self.n_segments
        self.tcn_channels = args.tcn_channels

        self.BiMap_Block      = self._make_BiMap_block(len(self.dims)//2)
        self.LogEig           = LogEig()

        self.tcn_width        =  self.n_bands
        self.Temporal_Block   = nn.Conv2d(1, self.tcn_channels, (self.kernel_size, self.tcn_width*self.dims[-1]**2), stride=(1, self.dims[-1]**2), padding=0)
        
        if self._mlp:
            self.Classifier = nn.Sequential(
            nn.Linear(self.tcn_channels, self.tcn_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.tcn_channels, self.tcn_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.tcn_channels, classes),
            nn.Softmax(dim = 1)
            )
        else:
            self.Classifier = nn.Sequential(
                nn.Linear(self.tcn_channels, classes),
                nn.Softmax(dim = 1)
            )
    
    def _make_BiMap_block(self, layer_num):
        layers = []

        if layer_num > 1:
          for i in range(layer_num-1):
            dim_in, dim_out = self.dims[2*i], self.dims[2*i+1]
            layers.append(BiMapGroups(self.channel_in, 1, dim_in, dim_out)) # in shape: (batch_size, band*seg, 1, dim_in, dim_in)
            layers.append(ReEig())
            layers.append(Rearrange('e g c1 c2 -> e g 1 c1 c2'))
        
        dim_in, dim_out = self.dims[-2], self.dims[-1]
        layers.append(BiMapGroups(self.channel_in, 1, dim_in, dim_out))
        layers.append(DomainAdaptiveBatchNorm(dim_out, momentum=0.1, rotate=False, bias=True))
        layers.append(ReEig())
          
        return nn.Sequential(*layers)

    def freeze(self):
        for module in [self.BiMap_Block, self.Temporal_Block]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for module in [self.BiMap_Block, self.Temporal_Block]:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x, return_latent=False):
        x     = rearrange(x, 'e q b c1 c2 -> e (b q) 1 c1 c2')

        x_csp = self.BiMap_Block(x)

        x_log = self.LogEig(x_csp)

        # NCHW Format: (batch_size, window_num*band_num, c, c) ---> (batch_size, 1, window_num, band_num * c * c)
        x_vec = rearrange(x_log, 'e (b q) c1 c2 -> e 1 q (b c1 c2)', b=self.n_bands)

        x_vec = self.Temporal_Block(x_vec).reshape(x.shape[0], -1)

        y     = self.Classifier(x_vec)

        if return_latent:
            return y, x_vec
        else:
            return y
        
   
class CombinedConv(nn.Module):
    """
    Adopted from: https://github.com/braindecode/braindecode/blob/master/braindecode/models/modules.py
    """

    def __init__(
        self,
        in_chans,
        n_filters_time=40,
        n_filters_spat=40,
        filter_time_length=25,
        bias_time=True,
        bias_spat=True,
    ):
        super().__init__()
        self.bias_time = bias_time
        self.bias_spat = bias_spat
        self.conv_time = nn.Conv2d(
            1, n_filters_time, (filter_time_length, 1), bias=bias_time, stride=1
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_spat, (1, in_chans), bias=bias_spat, stride=1
        )

    def forward(self, x):
        # Merge time and spat weights
        combined_weight = (
            (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3))
            .sum(0)
            .unsqueeze(1)
        )

        # Calculate bias term
        if not self.bias_spat and not self.bias_time:
            bias = None
        else:
            bias = 0
            if self.bias_time:
                bias += (
                    self.conv_spat.weight.squeeze()
                    .sum(-1)
                    .mm(self.conv_time.bias.unsqueeze(-1))
                    .squeeze()
                )
            if self.bias_spat:
                bias += self.conv_spat.bias

        return torch.nn.functional.conv2d(x, weight=combined_weight, bias=bias, stride=(1, 1))


class Expression(nn.Module):
    """
    Adopted from: https://github.com/braindecode/braindecode/blob/master/braindecode/models/modules.py
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return self.__class__.__name__ + "(expression=%s) " % expression_str


class ShallowFBCSPNet(nn.Module):
    """
    Shallow ConvNet model from Schirrmeister et al. [1]. Adopted from https://github.com/braindecode/braindecode/blob/master/braindecode/models/shallow_fbcsp.py

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(self, args):
        super().__init__()
        
        self.n_chans = args.n_chans
        self.n_classes = args.n_classes
        self.n_times = args.n_times
        
        self.n_filters_time = args.n_filters_time
        self.filter_time_length = args.filter_time_length
        self.n_filters_spat = args.n_filters_spat
        self.pool_time_length = args.pool_time_length
        self.pool_time_stride = args.pool_time_stride
        self.batch_norm = args.batch_norm
        self.batch_norm_alpha = args.batch_norm_alpha
        self.drop_prob = args.drop_prob
        
        self.procs = nn.Sequential(
            Rearrange('e c t -> e 1 t c'),
            CombinedConv(
                in_chans=self.n_chans,
                n_filters_time=self.n_filters_time,
                n_filters_spat=self.n_filters_spat,
                filter_time_length=self.filter_time_length,
                bias_time=True,
                bias_spat=not self.batch_norm,
            ),
            nn.BatchNorm2d(
                self.n_filters_spat, momentum=self.batch_norm_alpha, affine=True
            ),
            Expression(ShallowFBCSPNet.square_),
            nn.AvgPool2d(
                kernel_size=(self.pool_time_length, 1),
                stride=(self.pool_time_stride, 1),
            ),
            Expression(ShallowFBCSPNet.safe_log_),
            nn.Dropout(p=self.drop_prob)
        )

        self.eval()
        self.final_conv_length = self.get_output_shape()[2]

        # Incorporating classification module and subsequent ones in one final layer
        self.out = nn.Sequential(
            nn.Conv2d(
                self.n_filters_spat,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True,
            ),
            nn.Softmax(dim=1),
            Expression(ShallowFBCSPNet.squeeze_final_output)
        )

        # Initialization
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias"):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def square_(x):
        return x * x

    @staticmethod
    def safe_log_(x, eps=1e-6):
        """Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
        return torch.log(torch.clamp(x, min=eps))
    
    @staticmethod
    def squeeze_final_output(x):
        assert x.size()[3] == 1
        x = x[:, :, :, 0]
        if x.size()[2] == 1:
            x = x[:, :, 0]
        return x
    
    def get_output_shape(self):
        with torch.inference_mode():
            return tuple(
                self.procs(
                    torch.zeros(
                        (1, self.n_chans, self.n_times),
                        dtype=next(self.parameters()).dtype,
                        device=next(self.parameters()).device,
                    )
                ).shape
            )
        
    def freeze(self):
        for param in self.procs.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.procs.parameters():
            param.requires_grad = True

    def forward(self, x, return_latent=False):
        x = self.procs(x)
        pred = self.out(x)

        if return_latent:
            return pred, torch.flatten(x, start_dim=1)
        else:
            return pred


#############################
### Util models
#############################

class Conv2dWithConstraint(nn.Conv2d):
    '''
    Within-group normalization for convolutional layers. I.e., depth-wise separable convolution proposed by Chollet et al. [1].
    Adopted from: https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py

    .. [1] Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 1251-1258). 
    '''
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class JointCCSA(nn.Module):
    def __init__(self, net):
        '''
        Jointly train a model using the classification and contrastive semantic alignment (CCSA) method proposed by Motiian et al. [1]. 

        [1] Saeid Motiian, Marco Piccirilli, Donald A Adjeroh, and Gianfranco Doretto. Unified deep supervised domain 
        adaptation and generalization. In Proceedings of the IEEE international conference on computer vision, pages 5715–5725, 2017.
        '''
        super(JointCCSA, self).__init__()
        self.net = net

    def freeze(self):
        self.net.freeze()

    def unfreeze(self):
        self.net.unfreeze()

    def LSA(self, X, ds, y, norm=True):
        '''
        Calculate the semantic alignment (SA) loss.

        Args:
            X (torch.Tensor): Latent features of shape (bs, d)
            y (torch.Tensor): Labels of shape (bs,)
            ds (torch.Tensor): Domain labels of shape (bs,)
            norm (bool, optional): Boolean flag indicating whether to normalize the loss. Defaults to True.

        Returns:
            torch.Tensor: Semantic alignment loss.
        '''
        dists = torch.cdist(X, X)

        n_compare = 0
        sa_loss = 0
        for c in torch.unique(y):
            for d1, d2 in itertools.combinations(torch.unique(ds), 2):
                mask1 = torch.bitwise_and(y==c, ds==d1)
                mask2 = torch.bitwise_and(y==c, ds==d2)
                sa_loss += 0.5 * torch.sum(dists[mask1][:, mask2])
                n_compare += 1

        if norm:
            sa_loss /= n_compare

        return sa_loss
    
    def LS(self, X, ds, y, norm=True):
        '''
        Calculate the separation (S) loss.

        Args:
            X (torch.Tensor): Latent features of shape (bs, d)
            y (torch.Tensor): Labels of shape (bs,)
            ds (torch.Tensor): Domain labels of shape (bs,)
            norm (bool, optional): Boolean flag indicating whether to normalize the loss. Defaults to True.

        Returns:
            torch.Tensor: Separation loss.
        '''
        margin = 1
        tensor_zero = torch.tensor(0., dtype=X.dtype, device=X.device)
        dists = torch.cdist(X, X)
        dists = torch.maximum(tensor_zero, margin-dists)

        n_compare = 0
        s_loss = 0
        for d1, d2 in itertools.combinations(torch.unique(ds), 2):
            for c1, c2 in itertools.combinations(torch.unique(y), 2):
                mask1 = torch.bitwise_and(y==c1, ds==d1)
                mask2 = torch.bitwise_and(y==c2, ds==d2)
                s_loss += 0.5 * torch.sum(dists[mask1][:, mask2])
                n_compare += 1

        if norm:
            s_loss /= n_compare
        
        return s_loss
    
    def forward(self, x, ds=None, y=None, norm=True):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Latent features of shape (bs, d).
            ds (torch.Tensor):  Domain labels of shape (bs,).
            y (torch.Tensor): Labels of shape (bs,)
            norm (bool, optional): Flag indicating whether to normalize the losses. Default is True.

        Returns:
            tuple: A tuple containing the following elements:
                - pred (torch.Tensor): Network predictions.
                - sa_loss (torch.Tensor or None): Semantic alignment loss (L_SA). None if y or ds is None.
                - s_loss (torch.Tensor or None): Separation loss (L_S). None if y or ds is None.
        """
        pred, latent = self.net(x, return_latent=True)
        if y is None or ds is None:
            return pred, None, None
        
        sa_loss = self.LSA(latent, ds, y, norm=norm)
        s_loss = self.LS(latent, ds, y, norm=norm)
        
        return pred, sa_loss, s_loss
    
    
class MultiTangent(BaseEstimator, TransformerMixin):
    '''
    Transform covariance matrices to tangent space and take the upper triangle as feature.
    Not a PyTorch neural network module. Works with numpy arrays.

    Parameters:
    - metric (str): The metric to use for tangent space transformation. Default is 'riemann'.
    - **kwargs: Additional keyword arguments to be passed to the mean_covariance function.

    Methods:
    - fit(X, y=None): Fit the MultiTangent transformer to the input data.
    - transform(X): Transform the input data to tangent space.

    Attributes:
    - reference_ (ndarray): The reference covariance matrix used for tangent space transformation.
    '''

    def __init__(self, metric='riemann', **kwargs) -> None:
        super().__init__()
        self.metric = metric
        self.kwargs = kwargs

    def fit(self, X, y=None):
        '''
        Fit the MultiTangent transformer to the input data.

        Parameters:
        - X (ndarray): Covariance matrices of shape (n_epochs, n_segments, n_ch, n_ch).
        - y (ndarray or None): Target values. Ignored in this implementation.

        Returns:
        - self (MultiTangent): The fitted MultiTangent transformer.
        '''
        out = rearrange(X, 'e q c1 c2 -> (e q) c1 c2')
        self.reference_ = mean_covariance(out, self.metric, **self.kwargs)
        return self

    def transform(self, X):
        '''
        Transform the input data to tangent space.

        Parameters:
        - X (ndarray): Covariance matrices of shape (n_epochs, n_segments, n_ch, n_ch).

        Returns:
        - out (ndarray): Transformed data in tangent space.
        '''
        n_epochs, n_segments, n_ch, _ = X.shape
        out = rearrange(X, 'e q c1 c2 -> (e q) c1 c2')
        out = tangent_space(out, self.reference_, metric=self.metric)
        return rearrange(out, '(e q) f -> e (q f)', q=n_segments)
    
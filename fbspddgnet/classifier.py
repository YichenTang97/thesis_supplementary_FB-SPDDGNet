import os
import torch

import os.path as op
import numpy as np
import torch.utils.data as Data

from sklearn.base import BaseEstimator, ClassifierMixin

from .models import FB_SPDDGBN
from .eval.train import *
from .dataset.GetDataSet import toDataloader, splitDataset

dtype = torch.float32

class FB_SPDDGNet_Classifier(ClassifierMixin, BaseEstimator):
    '''
    FB_SPDDGNet_Classifier is a classifier (sklearn API) based on the FB-SPDDGNet model.

    Parameters:
    - args (object): Arguments for the FB-SPDDGNet model.
    - source_domains (list): List of source domains.
    - target_domains (list): List of target domains.
    - rotate (bool, optional): Whether to apply rotation step for domain generalisation (see FB-SPDDGNet). Default is True.
    - bias (bool, optional): Whether to apply parallel shifting step (bias) for domain generalisation (see FB-SPDDGNet). Default is True.
    - parallel (bool, optional): Whether to use parallel computation for different domains (see FB-SPDDGNet). Default is True.
    - gpu (bool, optional): Whether to use GPU for computation. Default is False.
    - seed (int, optional): Random seed for reproducibility. Default is 42.
    - save_folder (str, optional): Folder to save model states. Default is 'saved_states'.
    - save_name (str, optional): Name of the saved model. Default is 'FB-SPDDGNet'.
    - verbose (int, optional): Verbosity level. Default is 0.

    Attributes:
    - net_ (FB_SPDDGBN): FB-SPDDGBN model instance.

    Methods:
    - fit(X, y, d, dataset=None, val_ratio=0.2, epochs=100, batch_size=700, lr=0.01, weight_decay=1e-4, loss_lambdas=[0., 0.1, 0.1], checkpoints=[]): Fits the model to the training data.
    - fine_tune(X, y, d, dataset=None, n_karcher_steps=40, epochs=100, lr=0.001, weight_decay=1e-4, loss_lambdas=[0., 0.1], checkpoints=[], train_checkpoint=None): Fine-tunes the model.
    - predict_proba(X, d, dataset=None, batch_size=100, finetune_checkpoint=None): Predicts class probabilities for the input samples.
    - predict(X, d, dataset=None, batch_size=100, finetune_checkpoint=None): Predicts class labels for the input samples.

    See also:
    - FB_SPDDGBN: FB-SPDDGNet model.

    Sample usage:
    ```python
    import os.path as op
    from omegaconf import OmegaConf
    from fbspddgnet import FB_SPDDGNet_Classifier

    # Define the model arguments
    args_s = f"""
    n_classes: 5
    n_ch: 64
    norm_momentum: 0.1
    n_bands: 6
    n_segments: 4
    overlap: 0
    conv_ch_1: 48
    conv_ch_2: 32
    conv_t: 16
    bi_ho_1: 6
    bi_no_1: 16
    bi_ho_2: 6
    bi_no_2: 8
    """
    args = OmegaConf.create(args_s)

    # Or load the model arguments from the fb_spddgnet.yaml file
    proj_dir = ... # path to the project directory where fbspddgnet is located
    args = OmegaConf.load(op.join(proj_dir, 'fbspddgnet', 'confs', 'fb_spddgnet.yaml'))

    
    # Define the source and target domains
    source_domains = ['P1', 'P2', 'P3']
    target_domains = ['P4', 'P5']

    # Load the data
    # X is the EEG data after been band-pass filtered using a filter bank, y is the class labels, and d is the domain labels (must be from source_domains and target_domains)
    # X has shape (n_samples, n_frequency_bands, n_channels, n_timestamps), y has shape (n_samples,), d has shape (n_samples,)
    X, y, d = ... # training data, 
    X_finetune, y_finetune, d_finetune = ... # fine-tuning data
    X_test, y_test, d_test = ... # test data

    # Create the classifier
    clf = FB_SPDDGNet_Classifier(args, source_domains, target_domains, rotate=True, bias=True, parallel=True, gpu=False, seed=42, save_folder='saved_states', save_name='FB-SPDDGNet', verbose=1)

    # Fit the classifier
    clf.fit(X, y, d, X_finetune, y_finetune, d_finetune, dataset=None, val_ratio=0.2, epochs=100, batch_size=700, lr=0.01, weight_decay=1e-4, loss_lambdas=[0., 0.1, 0.1], checkpoints=[])
    clf.fine_tune(X_finetune, y_finetune, d_finetune, dataset=None, n_karcher_steps=40, epochs=100, lr=0.001, weight_decay=1e-4, loss_lambdas=[0., 0.1], checkpoints=[], train_checkpoint=None) # fine-tune is required for target domain adaptation
    preds = clf.predict(X_test, d_test, dataset=None, batch_size=100, finetune_checkpoint=None)
    probas = clf.predict_proba(X_test, d_test, dataset=None, batch_size=100, finetune_checkpoint=None)

    acc = (preds == y_test).mean()
    print(f'Accuracy: {acc}')
    ```
    '''

    def __init__(self, args, source_domains, target_domains, rotate=True, bias=True, parallel=True, gpu=False, seed=42, 
                 save_folder='saved_states', save_name='FB-SPDDGNet', verbose=0):
        self.args = args
        self.source_domains = list(source_domains)
        self.target_domains = list(target_domains)
        self.rotate = rotate
        self.bias = bias
        self.parallel = parallel
        self.gpu = gpu
        self.seed = seed
        self.save_folder = save_folder
        self.save_name = save_name
        self.verbose = verbose
        self.net_ : FB_SPDDGBN = FB_SPDDGBN(args, self.source_domains, self.target_domains, rotate=rotate, bias=bias, parallel=parallel)

        if not op.exists(save_folder):
            os.makedirs(save_folder)

    def _to_dataset(self, X, y, d):
        X = torch.as_tensor(X, dtype=dtype)
        y = torch.as_tensor(y, dtype=torch.long)
        d = torch.as_tensor(d, dtype=torch.long)
        return Data.TensorDataset(X, y, d)

    def fit(self, X, y, d, dataset=None, val_ratio=0.2, epochs=100, batch_size=700, lr=0.01, weight_decay=1e-4, loss_lambdas=[1.0, 0.1, 0.1], checkpoints=[]):
        """
        Fits the classifier to the given data. The RADAM optimizer [1] is used for training. The best state is saved based on the validation loss.

        Args:
            X (array-like): The input features. Ignored if dataset is provided.
            y (array-like): The target labels. Ignored if dataset is provided.
            d (array-like): The domain labels. Ignored if dataset is provided.
            dataset (Dataset, optional): The dataset object. If not provided, it will be created from X, y, and d.
            val_ratio (float, optional): The ratio of validation set size to the total dataset size. Default is 0.2.
            epochs (int, optional): The number of training epochs/iterations. Default is 100.
            batch_size (int, optional): The batch size for training. Default is 700.
            lr (float, optional): The learning rate for training. Default is 0.01.
            weight_decay (float, optional): The weight decay for training. Default is 1e-4.
            loss_lambdas (list, optional): The list of loss lambdas (see eval.train.trainNetwork and FB-SPDDGNet). Default is [1.0, 0.1, 0.1].
            checkpoints (list, optional): The list of checkpoints to save during training. Default is [].

        Returns:
            self: The fitted classifier object.

        References:
        [1] Gary Bécigneul and Octavian-Eugen Ganea. Riemannian adaptive optimization methods. arXiv preprint arXiv:1810.00760, 2018.672
        """
        if dataset is None:
            dataset = self._to_dataset(X, y, d)

        if self.verbose:
            print(f'Splitting dataset into {100*(1-val_ratio)}% training and {100*val_ratio}% validation sets')
        val_size = int(val_ratio * len(dataset))
        train_loader, val_loader = splitDataset(dataset, val_size=val_size, seed=self.seed)
        train_loader = toDataloader(train_loader, batch_size, shuffle=True, gpu=self.gpu)
        val_loader = toDataloader(val_loader, val_size, shuffle=False, gpu=self.gpu)
        
        if self.gpu:
            self.net_ = self.net_.cuda()
        
        self.net_ = trainNetwork(self.net_, train_loader, val_loader, iterations=epochs, lr=lr, wd=weight_decay, 
                                 loss_lambdas=loss_lambdas, gpu=self.gpu, folder=self.save_folder, name=self.save_name, 
                                 checkpoints=checkpoints, verbose=self.verbose)
        self.net_.eval()

        return self
    
    def fine_tune(self, X, y, d, dataset=None, n_karcher_steps=40, epochs=100, lr=0.001, weight_decay=1e-4, loss_lambdas=[1.0, 0.1], checkpoints=[], train_checkpoint=None):
        """
        Fine-tunes the network.

        Args:
            X (array-like): Input data. Ignored if dataset is provided.
            y (array-like): Target labels. Ignored if dataset is provided.
            d (array-like): Domain labels. Ignored if dataset is provided.
            dataset (Dataset, optional): Custom dataset object. If not provided, a dataset will be created from X, y, and d.
            n_karcher_steps (int, optional): Number of Karcher steps for domain adaptation on target domains. Default is 40.
            epochs (int, optional): Number of epochs for fine-tuning. Default is 100.
            lr (float, optional): Learning rate for fine-tuning. Default is 0.001.
            weight_decay (float, optional): Weight decay for fine-tuning. Default is 1e-4.
            loss_lambdas (list, optional): List of loss lambdas for fine-tuning (see eval.train.fineTuneNetwork and FB-SPDDGNet). Default is [1.0, 0.1].
            checkpoints (list, optional): List of checkpoints to save during fine-tuning. Default is an empty list.
            train_checkpoint (int, optional): The checkpoint before which to load the trained model state. Default is None, where the overall best state during training will be loaded.

        Returns:
            self: The fine-tuned classifier object.

        """
        if dataset is None:
            dataset = self._to_dataset(X, y, d)
        loader = toDataloader(dataset, len(dataset), shuffle=True, gpu=self.gpu)

        x_t, _, d_t = next(iter(loader))
        if self.gpu:
            x_t = x_t.to('cuda:0')
            d_t = d_t.to('cuda:0')

        self.net_ = get_trained_model(self.save_folder, self.save_name, checkpoint=train_checkpoint, gpu=self.gpu)
        self.net_.freeze()
        self.net_.domain_adapt(x_t, d_t, k=n_karcher_steps)
        self.net_ = fineTuneNetwork(self.net_, loader, calib_iter=epochs, lr=lr, wd=weight_decay, 
                                    loss_lambdas=loss_lambdas, gpu=self.gpu, checkpoints=checkpoints,
                                    folder=self.save_folder, name=self.save_name, verbose=self.verbose, on_source=False)
        self.net_.unfreeze()
        self.net_.eval()
        self.finetune_iter_ = epochs
        
        return self

    def predict_proba(self, X, d, dataset=None, batch_size=100, finetune_checkpoint=None):
        """
        Predicts the class probabilities for the given input samples.

        Parameters:
        - X (array-like): Input samples. Ignored if dataset is provided.
        - d (array-like): Input labels. Ignored if dataset is provided.
        - dataset (Dataset, optional): Custom dataset object. If not provided, a dataset will be created from X and d.
        - batch_size (int, optional): The batch size for prediction. Default is 100.
        - finetune_checkpoint (int, optional): The checkpoint iteration for which to load the finetuned model. If not provided, it will use the complete fine-tuned model.

        Returns:
        - Array of predicted class probabilities for each input sample.
        """
        if dataset is None:
            dataset = self._to_dataset(X, np.zeros(X.shape[0]), d)
        loader = toDataloader(dataset, batch_size, shuffle=False, gpu=self.gpu)

        if finetune_checkpoint is None:
            finetune_checkpoint = self.finetune_iter_

        self.net_ = get_finetuned_model(self.save_folder, self.save_name, iter=finetune_checkpoint, gpu=self.gpu)

        self.net_.eval()
        all_preds = []
        for xb, _, db in loader:
            if self.gpu:
                xb = xb.to('cuda:0')
                db = db.to('cuda:0')
            with torch.no_grad():
                pred, _, _ = self.net_(xb, db, None, on_source=False) # softmaxed predictions
            all_preds.append(pred)
        
        all_preds = torch.cat(all_preds, dim=0)
        return all_preds.cpu().detach().numpy()
    
    def predict(self, X, d, dataset=None, batch_size=100, finetune_checkpoint=None):
        """
        Predicts the class labels for the given input samples.

        Parameters:
        - X (array-like): Input samples. Ignored if dataset is provided.
        - d (array-like): Input labels. Ignored if dataset is provided.
        - dataset (Dataset, optional): Custom dataset object. If not provided, a dataset will be created from X and d.
        - batch_size (int, optional): The batch size for prediction. Default is 100.
        - finetune_checkpoint (int, optional): The checkpoint iteration for which to load the finetuned model. If not provided, it will use the complete fine-tuned model.

        Returns:
        - Array of predicted class labels for each input sample.
        """
        probas = self.predict_proba(X, d, dataset=dataset, batch_size=batch_size, finetune_checkpoint=finetune_checkpoint)
        return np.argmax(probas, axis=1)

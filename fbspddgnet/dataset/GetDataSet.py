import warnings
import torch
import torch.utils.data as Data
import numpy as np
import os.path as op

from einops import rearrange
from enum import Enum
from sklearn.model_selection import train_test_split

from ..utils.modules import E2R

dtype = torch.float32

class DataSet(Enum):
    SPEECH = 1
    OVERT = 2

class DataMode(Enum):
    TRAIN = 1
    TEST = 2
    CALIB = 3

participants_pool = {
    DataSet.SPEECH: np.arange(1,17), # [1-16]
    DataSet.OVERT: np.arange(1,15), # [1-14]
}

# labels:
# SPEECH, OVERT - 'anger': 0, 'happiness': 1, 'neutral': 2, 'pleasure': 3, 'sadness': 4

def preprocess_data(x, y, d, pre_split=False, segments=4, overlap=0, gpu=False, verbose=False):
    """
    Preprocesses the input data for training or inference.

    Args:
        x (numpy.ndarray or torch.tensor): The input data.
        y (numpy.ndarray or torch.tensor): The target labels.
        d (numpy.ndarray or torch.tensor): The domain labels.
        pre_split (bool, optional): Whether to split the input data into temporal segments. Defaults to False.
        segments (int, optional): The number of segments to split the input data into. Defaults to 4.
        overlap (int, optional): The overlap ratio between segments. Defaults to 0.
        gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
        verbose (bool, optional): Whether to print the shape of the preprocessed data. Defaults to False.

    Returns:
        tuple: A tuple containing the preprocessed input data, target labels, and additional data.
    """
    x = torch.tensor(x, dtype=dtype)
    y = torch.tensor(y, dtype=torch.long)
    d = torch.tensor(d, dtype=torch.long)

    if gpu:
        x = x.to("cuda:0")
        y = y.to("cuda:0")
        d = d.to("cuda:0")

    if pre_split:
        e2r = E2R(segments=segments, overlap=overlap)
        x = e2r(x)

    if verbose:
        print(x.shape)

    return x, y, d


def getParticipantByBlock(sub: int, blocks, data_path: str, filter_bank=True):
    """
    Retrieve participant data by block.

    Args:
        sub (int): The participant ID.
        blocks (list): List of block numbers.
        data_path (str): The path to the data files.
        filter_bank (bool, optional): Whether to use filter bank data. Defaults to True.

    Returns:
        tuple: A tuple containing the concatenated data and labels arrays.
            - x (ndarray): The concatenated data array with shape (n_trials, [n_bands], n_ch, n_t).
            - y (ndarray): The concatenated labels array with shape (n_trials,).
    """
    fb = "_filter_bank" if filter_bank else ""
    xs = []
    ys = []
    for b in blocks:
        xs.append(np.load(op.join(data_path, f'P{sub}_block_{b}_epochs_data{fb}.npy'))) # (n_trials, [n_bands], n_ch, n_t), ~50 trials per block
        ys.append(np.load(op.join(data_path, f'P{sub}_block_{b}_labels.npy'))) # (n_trials, ), ~50 trials per block
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys)
    
    return x, y


def getParticipantData(dataset: DataSet, sub: int, data_path: str, mode: DataMode, filter_bank=False, calib_trial_per_cls=5):
    """
    Get the data for a specific participant in a dataset.

    Args:
        dataset (DataSet): The dataset to retrieve the data from.
        sub (int): The participant ID.
        data_path (str): The path to the data.
        mode (DataMode): The mode of the data (TRAIN, TEST, or CALIB).
        filter_bank (bool, optional): Whether to apply a filter bank. Defaults to False.
        calib_trial_per_cls (int, optional): The number of trials per class per participant for calibration mode. Defaults to 5.

    Returns:
        tuple: A tuple containing the input data (x) and the corresponding labels (y).
    """
    if dataset is DataSet.SPEECH:
        max_blocks = 4 if sub == 4 else 5 # participant 4 has only 4 blocks
    elif dataset is DataSet.OVERT:
        max_blocks = 6
    else:
        raise ValueError(f"Invalid dataset option: {dataset}")
    
    if mode is DataMode.TRAIN:
        x, y = getParticipantByBlock(sub, range(max_blocks), data_path, filter_bank=filter_bank)
    elif mode is DataMode.TEST:
        x, y = getParticipantByBlock(sub, range(1, max_blocks), data_path, filter_bank=filter_bank)
    elif mode is DataMode.CALIB:
        x, y = getParticipantByBlock(sub, [0], data_path, filter_bank=filter_bank)
        unique_classes, _ = np.unique(y, return_counts=True)
        selected_indices = [np.where(y == cls)[0][:calib_trial_per_cls] for cls in unique_classes]
        x = np.concatenate([x[indices] for indices in selected_indices], axis=0)
        y = np.concatenate([[cls]*len(indices) for cls, indices in zip(unique_classes, selected_indices)])
        if len(y) < calib_trial_per_cls*len(unique_classes):
            warnings.warn(f"Number of trials per class is less than {calib_trial_per_cls}")
    else:
        raise ValueError(f"Invalid data mode option: {mode}")

    return x, y

def splitDataset(dataset, val_size, seed=None):
    g = torch.Generator().manual_seed(seed) if not seed is None else torch.Generator()
    train_dataset, val_dataset = Data.random_split(dataset, [len(dataset) - val_size, val_size], generator=g)
    return train_dataset, val_dataset

def toDataloader(dataset, bs, shuffle, gpu=False):

    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = bs,
        shuffle = shuffle,
        num_workers = 0,
        pin_memory=True if not gpu else False
    )
    
    return loader


def getDataloaderOneFold(dataset: DataSet, test_participants, data_path: str, mode: DataMode, 
                         bs: int, filter_bank=True, pre_split=False, segments=4, overlap=0, 
                         val_ratio=0.2, val_split_seed=42, val_one_batch=True, calib_trial_per_cls=5, 
                         gpu=False, verbose=False):
    """
    Get the dataloader for a single fold of the dataset (leave-N-participants-out framework).

    Args:
        dataset (DataSet): The dataset object.
        test_participants (list): List of participants to be used for testing.
        data_path (str): Path to the data.
        mode (DataMode): The mode of the data (TRAIN, TEST, or CALIB).
        bs (int): Batch size.
        filter_bank (bool, optional): Whether to use filter bank. Defaults to True.
        pre_split (bool, optional): Whether to pre-split the data. Defaults to False.
        segments (int, optional): Number of temporal segments to split the data into. Defaults to 4.
        overlap (int, optional): Overlap ratio between segments. Defaults to 0.
        val_ratio (float, optional): Ratio of validation data from source participants. Defaults to 0.2.
        val_split_seed (int, optional): Seed for random splitting of validation data. Defaults to 42.
        val_one_batch (bool, optional): Whether to use one batch for validation set. Defaults to True.
        calib_trial_per_cls (int, optional): Number of trials per class per participant for calibration. Defaults to 5.
        gpu (bool, optional): Whether to use GPU. Defaults to False.
        verbose (bool, optional): Whether to print shape of EEG data. Defaults to False.

    Returns:
        torch.utils.data.DataLoader or tuple: The dataloader for the data or a tuple of dataloaders for training and 
                                              validation data if mode is TRAIN and val_ratio is not zero.
    """
    subs = [s for s in participants_pool[dataset] if not s in test_participants] if mode is DataMode.TRAIN else test_participants

    x = []; y = []; d = []
    for s in subs:
        _x, _y = getParticipantData(dataset, s, data_path, mode, filter_bank=filter_bank, calib_trial_per_cls=calib_trial_per_cls)
        x.append(_x)
        y.append(_y)
        d.append(np.zeros(_x.shape[0])+s)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y)
    d = np.concatenate(d)

    dataset = Data.TensorDataset(*preprocess_data(x,y,d, pre_split=pre_split, segments=segments, overlap=overlap, gpu=gpu, verbose=verbose))
    if (not mode is DataMode.TRAIN) or val_ratio == 0:
        return toDataloader(dataset, bs, shuffle=True, gpu=gpu)
    else:
        val_size = int(val_ratio * len(dataset))
        train_dataset, val_dataset = splitDataset(dataset, val_size, seed=val_split_seed)
        train_loader = toDataloader(train_dataset, bs, shuffle=True, gpu=gpu)
        val_bs = val_size if val_one_batch else 1
        val_loader = toDataloader(val_dataset, val_bs, shuffle=False, gpu=gpu)
        return train_loader, val_loader


def getNumpyOneFold(dataset: DataSet, test_participants, data_path: str, mode: DataMode, 
                    val_ratio=0, random_state=42, filter_bank=True, spd=True, segments=4, overlap=0):
    """
    Get the numpy data for a single fold of the dataset (leave-N-participants-out framework).

    Args:
        dataset (DataSet): The dataset to retrieve the data from.
        test_participants (list): List of participants to use as test data.
        data_path (str): The path to the dataset.
        mode (DataMode): The mode of the data (TRAIN, TEST, or VAL).
        val_ratio (float, optional): The ratio of validation data to split from the training data. Defaults to 0.
        random_state (int, optional): The random state for train validation splitting. Defaults to 42.
        filter_bank (bool, optional): Whether to use filter bank features. Defaults to True.
        spd (bool, optional): Whether to use pre-computed SPD (Symmetric Positive Definite) features. Defaults to True.
        segments (int, optional): The number of segments to divide the data into. Defaults to 4.
        overlap (int, optional): The overlap between segments. Defaults to 0.

    Returns:
        tuple: A tuple containing the numpy data and labels. If mode is TEST or val_ratio is 0, returns (x, y).
        Otherwise, returns (x_train, y_train, x_val, y_val).
    """
    assert not mode is DataMode.CALIB, "CALIB mode is not supported for getNumpyOneFold"
    subs = [s for s in participants_pool[dataset] if not s in test_participants] if mode is DataMode.TRAIN else test_participants
    
    x = []; y = []
    for s in subs:
        _x, _y = getParticipantData(dataset, s, data_path, mode=mode, filter_bank=filter_bank)
        x.append(_x)
        y.append(_y)
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y)

    if spd:
        x = torch.tensor(x, dtype=torch.float32)
        e2r = E2R(segments=segments, overlap=overlap)
        x = e2r(x)
        x = x.numpy()
        x = rearrange(x, 'e q b c1 c2 -> e (b q) c1 c2')
    if (mode is DataMode.TEST) or val_ratio == 0:
        return x, y
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_ratio, random_state=random_state)
        return x_train, y_train, x_val, y_val

# FB-SPDDGNet for EEG-based emotion vocalisation decoding (thesis chapter supplementary code)

This repository implements the proposed FB-SPDDGNet architecture for cross-participant electroencephalogram (EEG) based emotion decoding during overt and imagined emotional vocalisations. The FB-SPDDGNet comprises three blocks: a feature extraction (FE) block, a Riemannian geometry-based domain-specific domain generalisation (DG) layer, and a classification block, illustrated in Figure 1. The DG layer is noval. It contains domain-specific adaptation modules which perform centring, scaling, rotating, and parallel shifting (bias) of symmetric positive definite (SPD) matrices separately for each domain to match SPD matrix distributions. We see each participant as an individual domain and associate one adaptation module to each participant. Detailes will be publically available when the thesis chapter (paper) is released/published.

![alt text](network_illustration.svg)

> **Figure 1**: The proposed FB-SPDDGNet architecture consists of a EEG feature extraction (FE) block, a Riemannian geometry-based domain-specific domain generalisation (DG) layer, and a classification block. Panel (A) shows an illustration for the overall structure of FB-SPDDGNet. Given a trial belonging to a specific participant, the adaptation module for this participant (e.g., $\text{D}_1$) is activated while the rest (e.g., $\text{D}_2$, $\cdots$, $\text{D}_N$) are not used. Panel (B) shows an illustration for the FE block. The FE block comprises a filter-band of six bandpass filters, followed by two convolutional layers for extracting spatio-temporal EEG features and a temporal segmentation step. Then, the FE block computes covariance matrices from the temporal segments and applies a BiMap layer [1] to generate more compact and discriminative SPD matrices on a Riemannian manifold. Panel (C) illustrates the classification block, comprising ReEig and BiMap layers [1] for learning discriminative mappings of SPD matrices, a LogEig layer [1] for mapping the matrices onto a flat tangent space, a matrix vectorisation step, and finally a fully connected linear layer for multi-class classification.

## Evaluation against baseline methods

### Experiments

We evaluated the classification methods separately using EEG datasets collected for an overt and an imagined emotional vocalisation experiment. Briefly, the participants listened to emotional vocalisations in five emotion categories (i.e., anger, happiness, neutral, pleasure, sadness), selected from the Montreal Affective Voices (MAV) corpus [2]. Then, the participants recognised the emotion and produced vocalisations in the same emotion either overtly or by imagining. Fourteen participants (12 males and 2 females; 26.9±2.75 years) and sixteen participants (7 males and 9 females; 29.4±14.56 years) took part in the overt and imagined vocalisation experiments, respectively. We collected 64-channel EEGs for all participants positioned under an international 10-10 system, at a sampling rate of 2048 samples per second. We applied a band-pass filter between 1 and 100 Hz and a notch-filter at 50 Hz and 100 Hz, re-referenced the EEG recordings to the average across all channels, removed EOG contamination using the FastICA algorithm, truncated the EEG recordings into four-second epochs starting from the vocalisation onsets, and down-sampled the data to 256 samples per second. All trials in which the participants incorrectly recognised the emotion of the vocalisation were removed from further analysis.

### Baseline methods and classification framework

We evaluated the proposed FB-SPDDGNet against several baseline methods including non-neural network-based filter bank tangent space logistic regression (FBTSLR) [3], filter bank common spatial pattern (FBCSP) [4], as well as neural network-based EEGNet [5], ShallowFBCSPNet [6], and Tensor-CSPNet [7] models. For neural network-based methods, we also applied a DG method, classification and contrastive semantic alignment (CCSA) [8], for comparison against the proposed DG layer. We used a leave-N-participants-out classification framework (N=2). In each classification fold, we selected two participants as targets and the rest of the participants as sources. We randomly splitted the source participants' data into training (80%) and validation (20%) sets. From each target participant, we also select 5 trials per class as a calibration set to be used for fine-tune the trained models. We used the EEG data recorded from second block onwards from the target participants as the testing set. We trained the classifiers on the training set and used the validation set for hyper-parameter tuning (non-neural network methods) or selecting the best training iteration with the lowest validation loss (neural network). We fine-tuned the last layer (classification layer) of the neural networks using the calibration set. Finally, we evaluated the methods on the testing set using accuracy score. We repeated the above process until all participants were selected as target once. This resulted in 7 folds (14 participants / 2 participant per fold) for the overt and 8 folds (16/2) for the imagined vocalisation dataset. We repeated the evaluations for all neural networks for 10 times with random parameter initialisations for a more accurate evaluation result.

## Results

Table 1 summarises the cross-participant classification accuracies and the standard deviations across classification folds (leave-N-participants-out framework, N=2) for FB-SPDDGNet and baseline methods.

>**Table 1**: The proposed FB-SPDDGNet achieved the highest classification accuracies compared to multiple baseline classification methods and a DG algorithm (i.e., CCSA). All methods were evaluated under a leave-N-participants-out classification framework (N=2). All neural network models were evaluated ten times with random parameter initialisation. This table summarises the averaged classification accuracies (%) and the standard deviations across folds (14 participants / 2 participants per fold = 7 folds for the overt vocalisation dataset, and 16/2=8 folds for the imagined vocalisation dataset) for all methods in comparison. Chance=20% for both datasets.

| Method                 | Overt vocalisation     | Imagined vocalisation  |
|------------------------|------------------------|------------------------|
| FBTSLR                 | 31.80 ± 5.02           | 24.24 ± 4.26           |
| FBCSP                  | 29.12 ± 4.69           | 23.84 ± 3.81           |
| EEGNet                 | 43.90 ± 6.77           | 21.55 ± 2.28           |
| ShallowFBCSPNet        | 46.79 ± 7.74           | 23.58 ± 5.24           |
| Tensor-CSPNet          | 42.87 ± 4.67           | 25.95 ± 5.32           |
| EEGNet + CCSA          | 43.73 ± 7.23           | 22.49 ± 3.10           |
| ShallowFBCSPNet + CCSA | 41.88 ± 5.67           | 22.67 ± 3.56           |
| Tensor-CSPNet + CCSA   | 41.23 ± 4.51           | 24.79 ± 3.87           |
| **FB-SPDDGNet (proposed)** | **49.47 ± 9.00**      | **26.74 ± 4.63**       |

<br>

# Requirements

The evaluated were completed on python 3.11.8 with the following libraries and versions:

```
torch==2.2.2+cu118
numpy==1.25.2
scikit-learn==1.4.2
pyriemann==0.6
einops==0.7.0
omegaconf==2.3.0
tqdm==4.66.2
```

The `fbspddgnet` library can be dynamically loaded using `importlib`:

```python
import sys
import importlib.util
import os.path as op

def import_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

proj_dir = ... # the project directory where the "fbspddgnet" folder is located
import_local("fbspddgnet", op.join(proj_dir, "fbspddgnet", "__init__.py"))

from fbspddgnet.models import FB_SPDDGNet

net = FB_SPDDGBN(...)
```

# Sample usage of package

This repository also implemented an scikit-learn API compatible class for classifying EEG signals using FB-SPDDGNet. The EEG data need to be band-pass filtered using a filter bank before using the classifier. Here is a sample code snippet:

```python
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

# Define the source and target domains
source_domains = [d+1 for d in range(12)] # participants 'P1' to 'P12'
target_domains = [13, 14] # Participants 'P13' and 'P14'

# Load the data
# X is the EEG data after been band-pass filtered using a filter bank, y is the class labels, 
# and d is the domain labels (must be from source_domains and target_domains)
# X has shape (n_samples, n_frequency_bands, n_channels, n_timestamps), y has shape (n_samples,), d has shape (n_samples,)
X, y, d = ... # training data, 
X_finetune, y_finetune, d_finetune = ... # fine-tuning data
X_test, y_test, d_test = ... # test data

# Create the classifier
clf = FB_SPDDGNet_Classifier(args, source_domains, target_domains, rotate=True, bias=True, 
                             parallel=True, gpu=False, seed=42, save_folder='saved_states', save_name='FB-SPDDGNet', verbose=1)

# Fit the classifier on source participants
clf.fit(X, y, d, dataset=None, val_ratio=0.2, epochs=100, batch_size=700, 
        lr=0.01, weight_decay=1e-4, loss_lambdas=[1.0, 0.1, 0.1], checkpoints=[])

# Fine-tune on target participants
clf.fine_tune(X_finetune, y_finetune, d_finetune, dataset=None, n_karcher_steps=40, 
              epochs=100, lr=0.001, weight_decay=1e-4, loss_lambdas=[1.0, 0.1], 
              checkpoints=[], train_checkpoint=None) # fine-tune is required for target domain adaptation

# Make predictions on target participants
preds = clf.predict(X_test, d_test, dataset=None, batch_size=100, finetune_checkpoint=None)
probas = clf.predict_proba(X_test, d_test, dataset=None, batch_size=100, finetune_checkpoint=None)

acc = (preds == y_test).mean()
print(f'Accuracy: {acc}')
```



# References
[1] Zhiwu Huang and Luc Van Gool. A Riemannian network for SPD matrix learning. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017. <br>
[2] Pascal Belin, Sarah Fillion-Bilodeau, and Frédéric Gosselin. The Montreal Affective Voices: A validated set of nonverbal affect bursts for research on auditory affective processing. Behavior research methods, 40(2):531–539, 2008. <br>
[3] Alexandre Barachant, Stéphane Bonnet, Marco Congedo, and Christian Jutten. Multiclass brain-computer interface classification by Riemannian geometry. IEEE Transactions on Biomedical Engineering, 59(4):920–928, 2011. <br>
[4] Kai Keng Ang, Zheng Yang Chin, Haihong Zhang, and Cuntai Guan. Filter bank common spatial pattern (FBCSP) in brain-computer interface. In 2008 IEEE international joint conference on neural networks (IEEE world congress on computational intelligence), pages 2390–2397. IEEE, 2008. <br>
[5] Vernon J Lawhern, Amelia J Solon, Nicholas R Waytowich, Stephen M Gordon, Chou P Hung, and Brent J Lance. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of Neural Engineering, 15(5):056013, 2018. <br>
[6] Robin Tibor Schirrmeister, Jost Tobias Springenberg, Lukas Dominique Josef Fiederer, Martin Glasstetter, Katharina Eggensperger, Michael Tangermann, Frank Hutter, Wolfram Burgard, and Tonio Ball. Deep learning with convolutional neural networks for EEG decoding and visualization. Human brain mapping, 38(11):5391–5420, 2017. <br>
[7] Ce Ju and Cuntai Guan. Tensor-cspnet: A novel geometric deep learning framework for motor imagery classification. IEEE Transactions on Neural Networks and Learning Systems, 34(12):10955–10969, 2022. <br>
[8] Saeid Motiian, Marco Piccirilli, Donald A Adjeroh, and Gianfranco Doretto. Unified deep supervised domain adaptation and generalization. In Proceedings of the IEEE international conference on computer vision, pages 5715–5725, 2017.

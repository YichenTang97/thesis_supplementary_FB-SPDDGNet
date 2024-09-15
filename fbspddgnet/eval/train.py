import os.path as op
import torch.nn as nn

from tqdm import tqdm

from ..models import *

def trainNetwork(net, trainloader, valloader, iterations=100, lr=0.01, wd=0.0001, loss_lambdas=[1.0, 0.1, 0.1], gpu=False, folder='results', name='model', checkpoints=[], verbose=False):
    """
    Trains the neural network. The training log is saved. The best model based on validation loss is saved as well as checkpoints at specified iterations.

    Args:
        net (torch.nn.Module): The neural network model to train.
        trainloader (torch.utils.data.DataLoader): The data loader for the training dataset. It should contain batches of data, labels and domain labels.
        valloader (torch.utils.data.DataLoader): The data loader for the validation dataset. It should contain batches of data, labels and domain labels.
        iterations (int, optional): The number of training iterations. Defaults to 100.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        wd (float, optional): The weight decay for the optimizer. Defaults to 0.0001.
        loss_lambdas (list, optional): The weights for the different loss terms (only used for FB-SPDDGNet and JointCCSA). Defaults to [1.0, 0.1, 0.1] for FB-SPDDGNet.
        gpu (bool, optional): Specifies whether to use GPU for training. Defaults to False.
        folder (str, optional): The folder to save the training results. Defaults to 'results' -- to be customised.
        name (str, optional): The name for the saved model. Defaults to 'model' -- to be customised.
        checkpoints (list, optional): The iterations at which to save checkpoints during training. Defaults to [].
        verbose (bool, optional): Specifies whether to print training progress. Defaults to False.

    Returns:
        torch.nn.Module: The trained neural network model.
    """
    if gpu:
        net.to("cuda:0")
    else:
        net.to("cpu")

    best_val_loss = float('inf')
    L = nn.NLLLoss()
    optimizer = geoopt.optim.RiemannianAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)

    fname = op.join(folder, name)
    log_f = f'{fname}-log.txt'
    open(log_f, 'w')

    with open(log_f, 'a') as f:
        iters = tqdm(range(iterations)) if verbose else range(iterations)
        for ite in iters:
            net.train()
            acc_tr = 0
            tr_len = 0
            for xb, yb, db in trainloader:
                if gpu:
                    xb = xb.to('cuda:0')
                    yb = yb.to('cuda:0')
                    db = db.to('cuda:0')
                tr_len += yb.shape[0]
                if type(net) is JointCCSA:
                    out, lsa, ls = net(xb, db, yb, norm=True)
                    out = torch.log(out)
                    # loss = lambda0 * L_class + lambda1 * L_SA + lambda2 * L_S
                    loss = loss_lambdas[0] * L(out, yb) + loss_lambdas[1] * lsa + loss_lambdas[2] * ls
                elif type(net) is FB_SPDDGBN:
                    out, dist, sim = net(xb, db, yb, on_source=True)
                    out = torch.log(out)
                    # loss = lambda0 * L_class + lambda1 * L_proto + lambda2 * L_sim
                    # L_proto = mean of distances between the SPD matrices and the prototypes within class, regardless of domain (for matching domain distributions)
                    # L_sim = mean of similarities between prototypes from different classes (for facilitating class separability)
                    loss = loss_lambdas[0] * L(out, yb) + loss_lambdas[1] * torch.mean(dist) + loss_lambdas[2] * torch.mean(sim)
                else:
                    out = net(xb)
                    out = torch.log(out)
                    loss = L(out, yb)
                acc_tr += (torch.max(out, 1).indices==yb).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            net.eval()

            val_acc, val_loss = testNetwork(net, valloader, gpu)
            
            log_str = f'\nIteration{ite+1}=====\n' \
                + f'train_loss:{loss.item():.4f} \t val_loss:{val_loss:.4f}\n' \
                + f'train_acc:{acc_tr/tr_len:.4f} \t val_acc:{val_acc:.4f}\n'
            f.write(log_str)
            if verbose:
                print(log_str, end='')
            
            # Save best model so far based on validation loss
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                f.write(f'Best model so far at iteration {ite+1}...\n')
                if verbose: print(f'Best model so far at iteration {ite+1}...')
                torch.save(net, f'{fname}-best_state.pth')
            
            # Save best state and current state at checkpoints
            if ite+1 in checkpoints:
                f.write(f'\n')
                f.write(f'Saving checkpoint at iteration {ite+1}...\n')
                if verbose: 
                    print()
                    print(f'Saving checkpoint at iteration {ite+1}...')
                torch.save(net, f'{fname}-iter{ite+1}.pth')
                torch.save(optimizer, f'{fname}-optimizer_iter{ite+1}.pth')
                best = torch.load(f'{fname}-best_state.pth')
                torch.save(best, f'{fname}-best_state_by_iter{ite+1}.pth')
                
        net.to("cpu")
        torch.save(net, f'{fname}-final_state.pth')
        torch.save(optimizer, f'{fname}-optimizer_iter{iterations}.pth')
        
    return net


def fineTuneNetwork(net, calibloader, calib_iter=100, lr=0.001, wd=0.0001, checkpoints=[], loss_lambdas=[1.0, 0.1], gpu=False, folder='results', name='model', verbose=False, **kwargs):
    """
    Finetune the network on new domains/participants.

    Args:
        net (torch.nn.Module): The network model to be fine-tuned.
        calibloader (torch.utils.data.DataLoader): The data loader for calibration data. It should contain a single batch or batches of data, labels and domain labels.
        calib_iter (int, optional): The number of calibration iterations. Defaults to 100.
        lr (float, optional): The learning rate for optimization. Defaults to 0.001.
        wd (float, optional): The weight decay for optimization. Defaults to 0.0001.
        checkpoints (list[int], optional): The iterations at which to save the model checkpoints. Defaults to [].
        loss_lambdas (list[float], optional): The weights for the loss terms (only used for FB-SPDDGNet). Defaults to [1.0, 0.1].
        gpu (bool, optional): Whether to use GPU for training. Defaults to False.
        folder (str, optional): The folder to save the training results. Defaults to 'results' -- to be customised.
        name (str, optional): The name for the saved model. Defaults to 'model' -- to be customised.
        **kwargs: Additional keyword arguments to be passed to the network model.

    Returns:
        torch.nn.Module: The fine-tuned network model.
    """
    if gpu:
        net.to("cuda:0")
    else:
        net.to("cpu")
    
    fname = op.join(folder, name)
    net.freeze()
    net.train()
    
    L = nn.NLLLoss()
    optimizer = geoopt.optim.RiemannianAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)

    for ite in tqdm(range(calib_iter)) if verbose else range(calib_iter):
        for xb, yb, db in calibloader:
            if gpu:
                xb = xb.to('cuda:0')
                yb = yb.to('cuda:0')
                db = db.to('cuda:0')

            if type(net) is JointCCSA:
                pred, _, _ = net(xb, None, None, **kwargs)
                loss = L(pred, yb) 
            elif type(net) is FB_SPDDGBN:
                pred, dist, _ = net(xb, db, yb, **kwargs)
                pred = torch.log(pred)
                loss = loss_lambdas[0] * L(pred, yb) + loss_lambdas[1] * torch.mean(dist)
            else:
                pred = net(xb, **kwargs)
                pred = torch.log(pred)
                loss = L(pred, yb) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ite+1 in checkpoints:
            torch.save(net, f'{fname}_finetuned_iter{ite+1}.pth')
            torch.save(optimizer, f'{fname}_optimizer_finetuned_iter{ite+1}.pth')
    
    net.eval()
    net.unfreeze()

    net.to("cpu")
    torch.save(net, f'{fname}_finetuned_iter{calib_iter}.pth')
    torch.save(optimizer, f'{fname}_optimizer_finetuned_iter{calib_iter}.pth')
    return net


def testNetwork(net, testloader, gpu=False, **kwargs):
    """
    Evaluate the performance of a network on a test dataset.

    Args:
        net (torch.nn.Module): The network to be evaluated.
        testloader (torch.utils.data.DataLoader): The test data loader. It should contain batches of data, labels and domain labels.
        gpu (bool, optional): Whether to use GPU for training. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the network.

    Returns:
        Tuple[float, float]: A tuple containing the accuracy and loss of the network.

    """
    if gpu:
        net.to("cuda:0")
    else:
        net.to("cpu")

    net.eval()
    L = nn.NLLLoss()
    all_preds = []
    all_labels = []
    for xb, yb, db in testloader:
        if gpu:
            xb = xb.to('cuda:0')
            yb = yb.to('cuda:0')
            db = db.to('cuda:0')
        with torch.no_grad():
            if type(net) is JointCCSA:
                pred, _, _ = net(xb, None, None, **kwargs)
            elif type(net) is FB_SPDDGBN:
                pred, _, _ = net(xb, db, None, **kwargs)
            else:
                pred = net(xb, **kwargs)
            
        all_preds.append(pred)
        all_labels.append(yb)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc = (torch.max(all_preds, 1).indices==all_labels).sum().item()

    all_preds = torch.log(all_preds)
    loss = L(all_preds, all_labels).item()
        
    return acc / len(all_labels), loss


def get_trained_model(folder : str, name : str, checkpoint=None, gpu=False):
    """
    Load a trained model from a file.

    Args:
        folder (str): The folder containing the saved model.
        name (str): The name of the saved model.
        checkpoint (int, optional): The iteration number of the checkpoint to load. Defaults to None.
        gpu (bool, optional): Whether to load the model to GPU. Defaults to False.

    Returns:
        torch.nn.Module: The loaded model.
    """
    fname = op.join(folder, name) 
    fname += '-best_state.pth' if checkpoint is None else f'-best_state_by_iter{checkpoint}.pth'
    model = torch.load(fname)
    if gpu:
        model.to("cuda:0")
    else:
        model.to("cpu")
    return model

def get_finetuned_model(folder : str, name : str, iter : int, gpu=False):
    """
    Load a fine-tuned model from a file.

    Args:
        folder (str): The folder containing the saved model.
        name (str): The name of the saved model.
        iter (int): The number of iterations used for fine-tuning.
        gpu (bool, optional): Whether to load the model to GPU. Defaults to False.

    Returns:
        torch.nn.Module: The loaded model.
    """
    fname = op.join(folder, name) + f'_finetuned_iter{iter}.pth'
    model = torch.load(fname)
    if gpu:
        model.to("cuda:0")
    else:
        model.to("cpu")
    return model

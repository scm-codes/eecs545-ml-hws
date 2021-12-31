# EECS 545 Fall 2021
import itertools
import os
import torch


def save_checkpoint(model, epoch, checkpoint_dir, stats):
    """
    Save model checkpoint.
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'stats': stats,
    }

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)


def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model, the current epoch, and training losses.
    """
    def get_epoch(cp):
        return int(cp.split('epoch=')[-1].split('.checkpoint.pth.tar')[0])

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]
    cp_files.sort(key=lambda x: get_epoch(x))

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception('Checkpoint not found')
        else:
            return model, 0, []

    # Find latest epoch
    epochs = [get_epoch(cp) for cp in cp_files]

    if not force:
        epochs = [0] + epochs
        print('Which epoch to load from? Choose from epochs below:')
        print(epochs)
        print('Enter 0 to train from scratch.')
        print(">> ", end='')
        inp_epoch = int(input())
        if inp_epoch not in epochs:
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print('Which epoch to load from? Choose from epochs below:')
        print(epochs)
        print(">> ", end='')
        inp_epoch = int(input())
        if inp_epoch not in epochs:
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        stats = checkpoint['stats']
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)".format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """
    Delete all checkpoints in directory.
    """
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")

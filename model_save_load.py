import os
import torch
import shutil

def save_checkpoint(state, ckpt_dir, name, is_best=True):
    """
    Save a copy of the model so that it can be loaded at a future
    date. This function is used when the model is being evaluated
    on the test data.

    If this model has reached the best validation accuracy thus
    far, a seperate file with the suffix `best` is created.
    """
    print("[*] Saving model to {}".format(ckpt_dir))

    filename = f'{name}_ckpt.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)

    if is_best:
        filename = f'{name}_model_best.pth.tar'
        shutil.copyfile(
            ckpt_path, os.path.join(ckpt_dir, filename)
        )


def load_checkpoint(model, ckpt_dir, name):
    """
    Load the best copy of a model. This is useful for 2 cases:

    - Resuming training with the most recent model checkpoint.
    - Loading the best validation model to evaluate on the test data.

    Params
    ------
    - best: if set to True, loads the best model. Use this if you want
      to evaluate your model on the test data. Else, set to False in
      which case the most recent version of the checkpoint is used.
    """
    print("[*] Loading model from {}".format(ckpt_dir))

    filename = name + '_ckpt.pth.tar'
    if False:
        filename = '_model_best.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state'])
    # load variables from checkpoint

    return model, ckpt['epoch'], ckpt['loss_beta'], ckpt['best_valid_acc']
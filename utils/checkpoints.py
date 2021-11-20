import torch
from datetime import datetime


def save_checkpoint(state, epoch):
    """
    Save model checkpoints
    :param epoch: epoch
    :param state:
    :return:
    """
    now = datetime.now()
    moment_date = now.strftime("%d_%m_%Y__%H_%M_%S")
    filename = "checkpoints/checkpoint_num_{}__{}.pth.tar".format(epoch, moment_date)
    print("==< Saving checkpoint >==")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """
    Load pre-trained model's checkpoints
    :param checkpoint: checkpoint file name
    :param model: model class
    :param optimizer: optimizer
    :return:
    """

    print("==< Loading checkpoint >==")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]

    return step

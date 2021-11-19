import torch
import torchvision.transforms as transforms
from PIL import Image


def save_checkpoint(state, filename, epoch):
    """
    Save model checkpoints
    :param epoch: epoch
    :param state:
    :param filename:
    :return:
    """
    filename = "checkpoint_{}_.pth.tar".format(epoch)
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

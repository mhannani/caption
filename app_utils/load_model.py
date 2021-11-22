import torch
from utils.checkpoints import load_checkpoint
from utils.models import Captioner

checkpoints_path = "checkpoints/checkpoint_num_39__21_11_2021__16_33_06.pth.tar"


def load_model():
    """
    Load pre-trained model checkpoint.
    :return:
    """

    # hyperparameters
    embed_size = 256
    hidden_size = 256
    vocabulary_size = 2339
    num_layer = 1

    # load the model class
    model = Captioner(embed_size, hidden_size, vocabulary_size, num_layer)

    # load the checkpoint
    model, _ = load_checkpoint(torch.load(checkpoints_path, map_location=torch.device('cpu')), model)

    # turn on evaluation mode for the model
    model.eval()

    return model


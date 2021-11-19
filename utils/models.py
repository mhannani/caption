import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    """
    The encoder model.
    """
    def __init__(self, embed_size, is_train=False):
        """
        The class constructor.
        :param embed_size: The size of the embedding [image representation]
        :param is_train: Whether to turn on the training or testing mode
        """

        super().__init__()

        self.is_train = is_train

        # use a pretrained model as our cnn by tweaking the last layer of the model
        # we gonna use the google's inception model
        self.inception = models.inception_v3()




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
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)

        # replace the last linear layer
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        # rectified Linear Unit activation function
        self.relu = nn.ReLU()

        # setting up DropOut layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        """
        The forward pass.
        :param images: array_like
            Batch of images
        :return: array_like
            The representation of images in [inception.fc.in_features, embed_size]
        """

        # extract the features from the images
        features = self.inception(images)

        # make the last linear layer trainable by enabling gradient
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.is_train

        return self.dropout(self.relu(features))
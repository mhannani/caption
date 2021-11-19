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

        self.fc = self.inception

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


class RNN(nn.Module):
    """
    The Decoder class based on LSTM
    """

    def __init__(self, embed_size, hidden_size, vocabulary_size, num_layers):
        """
        The class constructor of the Recurrent Neural Network
        :param embed_size: Integer
            The embedding size
        :param hidden_size: Integer
            The hidden cell size of the LSTM
        :param vocabulary_size: Integer
            The vocabulary size of the corpus
        :param num_layers: Integer
            The number of LSTM cells of the network
        """

        # call the init function for the super class
        super().__init__()

        # embedding layer
        self.embed = nn.Embedding(vocabulary_size, embed_size)

        # the lstm layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)

        # linear layer
        self.linear = nn.Linear(hidden_size, vocabulary_size)

        # dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
        Forward pass of the RNN network.
        :param features: array_like
            Multidimensional representation of images.
        :param captions: array_like
            Representation of captions
        :return: outputs from each time step of the LSTMs.
        """

        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs



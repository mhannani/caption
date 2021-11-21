import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from vocabulary import Vocabulary


class FlickrDataset(Dataset):
    """
    The Flickr Dataset loader class.
    """
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        """
        The class constructor.
        params
        :root_dir string
            Root directory of the dataset.
        :caption_file string
            The name of thebuild_vocabulary caption csv file.
        :transform: transforms class
            The transformation to apply to data while loading it.
        :freq_threshold integer
            The threshold of the frequency of the vocabulary.
        
        """

        self.root_dir = root_dir
        self.captions_file = captions_file
        self.transform = transform
        self.freq_threshold = freq_threshold
        # load the dataframe of captions and image filename
        self.df = pd.read_csv(self.captions_file)

        # get the images and captions
        self.images = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize and build the vocabulary
        self.vocabulary = Vocabulary(freq_threshold)
        self.vocabulary.build_vocabulary(self.captions.tolist())

    def __len__(self):
        """
        returns the length of the dataset.
        :return: integer
        """

        return len(self.df)

    def __getitem__(self, item):
        """
        Fetches data sample for the given item/key.
        :param item: integer
            The index of the sample to be retrieved.
        :return: a sample
        """

        # get the caption and the image for the current given index
        caption = self.captions[item]
        image_filename = self.images[item]

        # read the image from the path using the filename
        image = Image.open(os.path.join(self.root_dir, image_filename)).convert('RGB')

        # transform an image of a transformer was given
        if self.transform is not None:
            image = self.transform(image)

        # build caption in numerical representation
        # add the Start Of Sentence
        numeric_caption = [self.vocabulary.stoi["<SOS>"]]
        # numericlize the caption
        numeric_caption += self.vocabulary.numericalize(caption)
        # add the End Of Sentence
        numeric_caption.append(self.vocabulary.stoi["<EOS>"])

        return image, torch.tensor(numeric_caption)








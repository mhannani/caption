import torch
from torch.nn.utils.rnn import pad_sequence

class Equiponderater:
    """

    """

    def __init__(self, pad_value):
        """
        Class constructor.
        :param pad_index:
        """

        self.pad_value = pad_value

    def __call__(self, batch):
        # get the images from the batch
        images = [image[0].unsqueeze(0) for image in batch]

        # concatenate the retrieved images along the axis-0(batch number axis)
        images = torch.cat(images, dim=0)

        # get the caption from the batch
        captions = [caption[1] for caption in batch]

        # pad captions
        caption_padded = pad_sequence(captions, batch_first=False, padding_value=self.pad_value)

        return images, caption_padded

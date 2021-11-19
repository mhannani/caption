from utils.flickr_dataset import FlickrDataset
from utils.equiponderate import Equiponderater
from torch.utils.data import DataLoader


def data_loader(root_dir, caption_file,
                transform, batch_size=32,
                num_workers=16, shuffle=True,
                pin_memory=True):
    """

    :param root_dir: string
        Dataset path
    :param caption_file: string
        Caption filename
    :param transform: Transforms
        Transform data
    :param batch_size: Integer
        number of sample per batch
    :param num_workers: Integer
        Number of subprocesses to use for the data.
    :param shuffle: Boolean
        Whether to shuffle data at each epoch or not.
    :param pin_memory: Boolean
        The data loader will copy Tensors into CUDA pinned memory before returning them

    :return: DataLoader instance.
    """

    # load the dataset
    dataset = FlickrDataset(root_dir, caption_file, transform)
    # padding value
    pad_value = dataset.vocabulary.stoi["<PAD>"]

    # wrap an iterable around our dataset using DataLoader class
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Equiponderater(pad_value=pad_value)
    )

    return loader



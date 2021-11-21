from utils.data_loader import data_loader


def load_vocabulary(transform):
    """

    :return:
    """
    # constants
    root_dir = "Data/Images/train"
    caption_file ="Data/caption_train.csv"

    # get the data
    _, train_dataset = data_loader(root_dir, caption_file, transform, num_workers=6)

    vocabulary = train_dataset.vocabulary

    return vocabulary

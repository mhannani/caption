import warnings
import torch
import torch.optim as optim
from data_loader import data_loader
import pandas as pd
from PIL import Image
from checkpoints import load_checkpoint
from models import Captioner
import torchvision.transforms as transforms


def inference(model, dataset, transform, device, image_name=None, show_image=False):
    """

    :param model: the model class
    :param dataset: dataset
    :param image_name: image name withour extension
    :return: string
        genereted caption
    """
    if image_name is None:
        warnings.WarningMessage("No image name was given please... Using the default one.")
        image_name = "3637013_c675de7705"

    # transform the image
    transformed_image = transform(
        Image.open(f"../Data/Images/test/{image_name}.jpg").convert("RGB")).unsqueeze(0)

    # get ground-truth captions of the given image
    df = pd.read_csv("../Data/caption_test.csv")
    image_captions = df.loc[df['image'] == f"{image_name}.jpg"]['caption'].to_list()

    # summary
    print("===============================================================")
    print("|==================< ground-truth captions >==================|")

    for caption in image_captions:
        print(caption)

    print("|====================< generated caption >====================|")

    if show_image:
        image = Image.open("../Data/Images/test/335588286_f67ed8c9f9.jpg")
        image.show()

    print(dataset.vocabulary)
    print(" ".join(model.image_captioner(transformed_image.to(device), dataset.vocabulary, max_length=100)))


if __name__ == "__main__":

    # define the transformation
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # load the pretrained model
    # hyperparameters
    embed_size = 256
    hidden_size = 256
    vocabulary_size = 2339
    num_layer = 1
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Captioner(embed_size, hidden_size, vocabulary_size, num_layer).to(device)

    model, _ = load_checkpoint(torch.load("checkpoints/checkpoint_num_39__21_11_2021__16_33_06.pth.tar",
                                          map_location=torch.device('cpu')), model)

    # set up the model to evaluation mode
    model.eval()

    # get the data
    training_data, train_dataset = data_loader(root_dir="../Data/Images/train",
                                               caption_file="../Data/caption_train.csv",
                                               transform=transform, num_workers=6)

    inference(model, train_dataset, transform, device, image_name="335588286_f67ed8c9f9")

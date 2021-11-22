import warnings
import pandas as pd
from PIL import Image


def inference(model, dataset, transform, device, image_name=None, show_image=False):
    """
    Doing inference using pretrained model
    :param show_image:
    :param device:
    :param transform:
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
        Image.open(f"Data/Images/test/{image_name}.jpg").convert("RGB")).unsqueeze(0)

    # get ground-truth captions of the given image
    df = pd.read_csv("Data/caption_test.csv")
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

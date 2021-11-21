import torch
from . import load_transform_img
from .load_model import load_model
import torchvision.transforms as transforms
from .load_vocabulary import load_vocabulary


def generate_caption(image):
    """
    Generate a caption for the given image
    :param image: A preprocessed image.
    :return: string
        Caption generated
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transformation
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # transform the image
    transformed_img = load_transform_img(image, transform)

    # load the pretrained model
    model = load_model()

    # get the vocabulary
    vocabulary = load_vocabulary(transform)

    # generate caption
    caption = " ".join(model.image_captioner(transformed_img.to(device), vocabulary, max_length=100))

    return caption




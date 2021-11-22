import torch
from . import load_transform_img
from .load_model import load_model
import torchvision.transforms as transforms
from .load_vocabulary import load_vocabulary


def generate_caption(image, num_captions=50):
    """
    Generate a caption for the given image
    :param image: A preprocessed image.
    :param num_captions: Number of captions
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
    # print(vocabulary)

    # generate caption
    caption = " ".join(model.image_captioner(transformed_img.to(device), vocabulary, max_length=num_captions))

    # remove <SOS> and <EOS> symbols from caption
    cleaned_caption = caption.replace('<EOS>', '').replace('<SOS>', '')

    return cleaned_caption




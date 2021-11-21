from PIL import Image
import torchvision.transforms as transforms

# hyperparameters
#     embed_size = 256
#     hidden_size = 256
#     vocabulary_size = 2339
#     num_layer = 1
#     lr = 3e-4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transform_img(img_buffer):
    """
    Loads and transforms the given image.
    :param img_buffer: Memory buffer that contains the image.
    :return: Torch.Tensor
        The transformed image
    """

    image = Image.open(img_buffer).convert("RGB")

    # define the transformation
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # transform the image and add an extra dimension for the batch [1, 3, 300, 300]
    transformed_image = transform(image).unsqueeze(0)

    return transformed_image




from . import load_transform_img


def generate_caption(image):
    """
    Generate a caption for the given image
    :param image: A preprocessed image.
    :return: string
        Caption generated
    """

    transformed_img = load_transform_img(image)

    #
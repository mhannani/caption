import pandas as pd


def gt_captions(filename, as_df=True):
    """
    Get caption for the images
    :param filename: string
        Filename
    :param as_df: Boolean
        if True returns dataframe of captions if not return an 1d-array of captions
    :return: Dataframe
        Dataframe of ground truth captions
    """

    # get captions for the images
    df = pd.read_csv('./Data/caption_test.csv')
    image_captions = df.loc[df['image'] == f"{filename}"]['caption'].to_list()

    # captions dataframe
    captions_df = pd.DataFrame(image_captions, columns=['Captions'])

    if as_df:
        return captions_df

    return image_captions

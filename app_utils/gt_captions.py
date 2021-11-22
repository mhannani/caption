import pandas as pd


def gt_captions(filename):
    """
    Get caption for the images
    :param filename: string
        Filename
    :return: Dataframe
        Dataframe of ground truth captions
    """

    # get captions for the images
    df = pd.read_csv('./Data/caption_test.csv')
    image_captions = df.loc[df['image'] == f"{filename}"]['caption'].to_list()

    # captions dataframe
    captions_df = pd.DataFrame(image_captions, columns=['Captions'])
    return captions_df

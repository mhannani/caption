from .gt_captions import gt_captions


def blue_score(filename, pred_caption):
    """
    Calculate the BLUE [Bilingual Evaluation Understudy] score of the generated caption.
    :param filename: string
        Image filename; used to get its ground truth captions
    :param pred_caption: string
        The generated caption
    :return: double
        BLUE score value
    """

    # get list of ground-truth caption
    captions = gt_captions(filename, as_df=False)

    print(captions)





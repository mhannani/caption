from .gt_captions import gt_captions
from .preprocess_str import pre_process_sentence
from nltk.translate.bleu_score import sentence_bleu


def calculate_blue_score(filename, pred_caption, weights):
    """
    Calculate the BLUE [Bilingual Evaluation Understudy] score of the generated caption.
    :param filename: string
        Image filename; used to get its ground truth captions
    :param pred_caption: string
        The generated caption
    :param weights: tuple
        Weights of the BLUE.
    :return: double
        BLUE score value
    """

    # Get list of ground-truth caption
    captions = gt_captions(filename, as_df=False)

    # preprocess ground-truth caption
    preprocessed_captions = []
    for caption in captions:
        preprocessed_caption = pre_process_sentence(caption)
        preprocessed_captions.append(preprocessed_caption)

    # preprocess generated caption
    preprocessed_pred_caption = pre_process_sentence(pred_caption)

    print(weights)

    blue_score = sentence_bleu(preprocessed_captions, preprocessed_pred_caption, weights)

    return blue_score


import json
import pandas as pd
from utils.vocabulary import Vocabulary


def main():
    vocab = Vocabulary(5)
    vocab_path_json = "app_utils/vocab.json"
    captions_file = "Data/captions.txt"
    captions_list = pd.read_csv(captions_file)["caption"].tolist()
    vocab_dict = vocab.build_vocabulary(captions_list)
    print("Total vocabulary size: {}".format(len(vocab_dict)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path_json))

    # save the vocabulary as json
    with open(vocab_path_json, 'w') as f:
        json.dump(vocab_dict, f)


if __name__ == '__main__':
    main()

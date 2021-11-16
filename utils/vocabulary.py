class Vocabulary:
    """
    Build and
    """

    def __init__(self, freq_threshold):
        """
        The class constructor.
        :param freq_threshold: integer
            The minimum frequency threshold to include a token in the vocabulary.
        """

        # index to string dictionary convertor
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # string to index dictionary converter
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}




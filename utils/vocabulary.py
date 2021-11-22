import spacy
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    """
    Initialize and Build the vocabulary.
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

        self.freq_threshold = freq_threshold

    def __len__(self):
        """
        Returns the length of the vocabulary.
        :return:
        """
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        """
        Tokenize the given string
        :param text: string
            a caption_like string
        :return: liss
            List of tokens
        """

        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, list_of_sentences):
        """
        Build the vocabulary from list of tokenized captions
        if a token exceed the threshold then will be included
        in vocabulary, not otherwise.

        :param list_of_sentences: array_like
            list of all captions
        :return:
        """

        # dictionary of frequencies
        frequencies = {}

        # count the each token frequency in our corpus
        index = 4

        # loop through the sentences/caption
        for sentence in list_of_sentences:
            # loop through each token in current sentence/caption
            for token in self.tokenizer_eng(sentence):
                # if first time encountered
                if token not in frequencies:
                    frequencies[token] = 1
                # increment the counter
                else:
                    frequencies[token] += 1

                # if the frequency of the token exceed the threshold add it to vocabulary
                if frequencies[token] == self.freq_threshold:
                    self.stoi[token] = index
                    self.itos[index] = token
                    index += 1

        return self.itos

    def numericalize(self, text):
        """
        Takes a text and converts it into numerical value.
        :param text: string
            text to be numericalized
        :return: array_like
            array of number presenting the token numericlized
        """

        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

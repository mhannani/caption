import re


def generate_ngrams(sentence, n_grams):
    """
    Generate n-grams from the given sentence [string]
    :param sentence: string
        A sentence
    :param n_grams: integer
        Number of tokens
    :return: list of n-grams
    """

    # Lowercase the string
    sentence = sentence.lower()

    # replace non alphanumeric with spaces
    sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)

    # tokenize the sentence and remove empty
    tokens = [token for token in sentence.split(' ') if token != ""]

    # get ngrams
    ngrams = zip(*[tokens[i:] for i in range(n_grams)])

    # construct list of ngrams
    ngrams = [" ".join(ngram) for ngram in ngrams]

    return ngrams


if __name__ == "__main__":
    print(generate_ngrams("I go to school yesterday to meet a friend", 2))

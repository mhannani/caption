import re


def pre_process_sentence(sentence):
    """
    Clean up a given sentence.
    :param sentence: string
        A sentence
    :return: string 
        Cleaned sentence
    """

    # Lowercase the string
    sentence = sentence.lower()

    # replace non alphanumeric with spaces
    sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
    
    return sentence
    

if __name__ == "__main__":
    print(pre_process_sentence("I go to school yesterday to meet a friend."))

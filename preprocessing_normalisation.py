import re
import contractions
import pandas as pd
from num2words import num2words


def preprocess(text: str) -> str:

    def replace_contractions(text: str) -> str:
        """
        Function to replace contractions in a string of text.
        E.g. isn't -> is not
        """
        return contractions.fix(text)

    def remove_URL(text: str) -> str:
        """
        Function to remove URLs from a string of text.
        """
        return re.sub(r"http\S+", "", text)

    text = remove_URL(text)
    text = replace_contractions(text)

    return text


def normalise(text: str) -> str:

    def lowerise(text):
        """
        Function that makes all words in a string lowercase
        """
        return text.lower()

    def remove_punctuation(text):
        """
        Function that removes all punctuation from a string of text
        """
        # check if text is None or non string
        if pd.isna(text):
            return text
        return re.sub(r'[^\w\s]', '', text)

    def remove_common_words(text):
        """
        Function that removes words that are the most common in english,
        those that dont provide information on the text context.
        """
        common_words = ["the", "our", "and", "a", "of",
                        "in", "to", "an", "is", "that", "were"]
        if pd.isna(text):
            return text
        text = " ".join([
            word for word in text.split() if word not in common_words
        ])
        return text

    def numbers_to_words(text: str) -> str:
        """
        Function that changes numerically represented numbers to
        their words.
        E.g. 67 -> sixty-seven
        """
        if pd.isna(text):
            return text

        def replace_with_words(match):
            return num2words(int(match.group()))

        return re.sub(r'\d+', replace_with_words, str(text))

    text = lowerise(text)
    text = numbers_to_words(text)
    text = remove_punctuation(text)
    text = remove_common_words(text)

    return text

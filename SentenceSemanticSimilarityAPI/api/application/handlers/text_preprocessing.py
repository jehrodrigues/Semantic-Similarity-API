from bs4 import BeautifulSoup
import re


class TextPreprocessing(object):
    """
    Handles text pre-processing
    """

    def __init__(self, sentence: str):
        self._sentence = sentence

    def clear_text(self) -> str:
        # remove the html tags
        soup = BeautifulSoup(self._sentence, 'html.parser')
        clean_sentence = soup.get_text()

        # remove line break
        clean_sentence = clean_sentence.replace('\n', ' ')

        # remove duplicate spaces
        clean_sentence = re.sub(' +', ' ', clean_sentence)
        return clean_sentence

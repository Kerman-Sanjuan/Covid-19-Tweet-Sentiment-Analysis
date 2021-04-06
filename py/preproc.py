import re
import string
import nltk
nltk.data.path.append("./nltk_data/")
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
PUNCT_TO_REMOVE = string.punctuation
PUNCT_TO_REMOVE += 'â’'
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_punctuation(text):
    """custom function to remove the punctuation"""
    spaces = ''
    for i in range(len(PUNCT_TO_REMOVE)):
        spaces += ' '
    return text.translate(str.maketrans(PUNCT_TO_REMOVE, spaces))

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

def preprocess_text(text_input):
    # URLS
    text_input = remove_urls(text_input)

    # LOWERCASE
    text_input = text_input.lower()


    # PUNCT
    text_input = remove_punctuation(text_input)

    # REGEX
    text_input = re.sub('[^A-Za-z ]+', '', text_input)

    # LEMMANIZATION
    text_input = lemmatize_words(text_input)

    # SPELL
    text_input = correct_spellings(text_input)

    return text_input


if __name__ == "__main__":
   
    preprocess_text()



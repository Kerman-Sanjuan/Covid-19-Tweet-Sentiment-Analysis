import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

text_input = 'Teams ranked 7-10 will participate in the NBA Play-In Tournament after the regular season (May 18-21) to secure the final two spots in the Playoffs for each conference.'
print(text_input)

# URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
text_input = remove_urls(text_input)
print(text_input)

# LOWERCASE
text_input = text_input.lower()
print(text_input)

# STOPWORDS
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add('u')
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in set(stopwords.words('english'))])

text_input = remove_stopwords(text_input)
print(text_input)

# PUNCT
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    PUNCT_TO_REMOVE = string.punctuation
    PUNCT_TO_REMOVE += 'â’'
    spaces = ''
    for i in range(len(PUNCT_TO_REMOVE)):
        spaces += ' '
    return text.translate(str.maketrans(PUNCT_TO_REMOVE, spaces))
text_input = remove_punctuation(text_input)
print(text_input)

# REGEX
text_input = re.sub('[^A-Za-z ]+', '', text_input)
print(text_input)

# LEMMANIZATION
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

text_input = lemmatize_words(text_input)
print(text_input)

# SPELL
spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

text_input = correct_spellings(text_input)
print(text_input)

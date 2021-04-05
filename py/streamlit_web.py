import streamlit as st
from io import StringIO
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

global current_dir

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add('u')
PUNCT_TO_REMOVE = string.punctuation
PUNCT_TO_REMOVE += 'â’'
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in set(stopwords.words('english'))])


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

    # STOPWORDS
    text_input = remove_stopwords(text_input)

    # PUNCT
    text_input = remove_punctuation(text_input)

    # REGEX
    text_input = re.sub('[^A-Za-z ]+', '', text_input)

    # LEMMANIZATION
    text_input = lemmatize_words(text_input)

    # SPELL
    text_input = correct_spellings(text_input)

    return text_input


def check_raw_text(raw_text):
    raw_text = preprocess_text(raw_text)
    global current_dir
    current_dir = os.getcwd() + "/"
    if st.button("Make prediction"):
        if not str(raw_text):  # if the text is empty
            st.error("Empty Text")
        else:
            st.success("Text processed correctly")
            transf = vectorizer.transform([raw_text])
            pred_df = pd.DataFrame(transf.todense(), columns=vectorizer.get_feature_names())[df.columns]
            y = LR.predict(pred_df)
            if y[0] == 0:
                st.error("AGAINST COVID TWEET")
            elif y[0] == 1:
                st.warning("NEUTRAL COVID TWEET")
            elif y[0] == 2:
                st.success("FAVORABLE COVID TWEET")


df = pd.read_csv("../csv/best_attr.csv").drop('Sentiment', axis=1)
LR = pickle.load(open("../models/logistic_regression.pk", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pk", "rb"))
st.title("Natural Language Processing Web Application")
st.header("What type of upload method would you like to use?")

option = st.selectbox('Upload method', ('Raw Text', 'External File'))

if option == 'Raw Text':

    raw = st.text_area("Enter the tweet")
    check_raw_text(raw)

else:
    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        check_raw_text(string_data)

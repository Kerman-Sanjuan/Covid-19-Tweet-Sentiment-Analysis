import streamlit as st
from io import StringIO
import os
import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from preproc import remove_urls
from preproc import lemmatize_words
from preproc import correct_spellings
from preproc import remove_punctuation
from preproc import preprocess_text
from keras.models import load_model
import nltk
import time

global current_dir
global classifier_type
global text_input_method
global checkbox
global input_text

st.write("hola")

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
from preproc import remove_stop_words
from keras.models import load_model
import nltk
import time

global current_dir
global classifier_type
global text_input_method
global checkbox
global input_text


def fix_input_text(text):
	if str(input_text):
		raw_data_lines = text.split("\n")
		s = ""
		for i in raw_data_lines:
			if i != "\n":
				s = s + " " + str(i)
		return s.replace("\n"," ")


def predict(pred_df):
	st.subheader("RESULTS")
	if classifier_type == "LOGISTIC REGRESSION":
		y = LR.predict(pred_df)
		if y[0] == 0:
			st.error("Sentiment: NEGATIVE")
		elif y[0] == 1:
			st.warning("Sentiment: NEUTRAL")
		elif y[0] == 2:
			st.success("Sentiment: POSITIVE")
	else: #NEURAL NETWORK
		y = NN.predict(pred_df)
		for i in y:
			st.error("PERCENTAGE SENTIMENT: NEGATIVE -> %" + str(i[0]*100))
			st.warning("PERCENTAGE SENTIMENT: NEUTRAL -> %" + str(i[1]*100))
			st.success("PERCENTAGE SENTIMENT: POSITIVE -> %" + str(i[2]*100)) 

		st.subheader("PREDICTED SENTIMENT")
		v = np.argmax(y[0])
		if v == 0:
			st.error("Sentiment: NEGATIVE")
		elif v == 1:
			st.warning("Sentiment: NEUTRAL")
		elif v == 2:
			st.success("Sentiment: POSITIVE")
	
def print_prediction_legend():
	st.header("PREDICTION LEGEND:")
	st.success("GREEN IF RESULT EQUALS POSITIVE")
	st.warning("YELLOW IF RESULT EQUALS NEUTRAL")
	st.error("RED IF RESULT EQUALS NEGATIVE")	   

def print_preprocess_steps():
	if not input_text:
		st.warning("NO TWEET HAS BEEN WRITTEN")
	else:
		change = input_text
		st.header("PREPROCESS STEPS")

		st.subheader("URL REMOVAL") 	
		text = remove_urls(change)
		st.write(text)
		time.sleep(1.5)

		st.subheader("TEXT TO LOWERCASE")
		text = text.lower()
		st.write(text)
		time.sleep(1.5)

		st.subheader("REMOVE STOPWORDS")
		text = remove_stop_words(text)
		st.write(text)
		time.sleep(1.5)

		st.subheader("REMOVE PUNCTUATION")
		text = remove_punctuation(text)
		st.write(text)
		time.sleep(1.5)

		st.subheader("REMOVE EVERYTHING EXCEPT ALPHABET CHARACTERS")
		text = re.sub('[^A-Za-z ]+', '', text)
		st.write(text)
		time.sleep(1.5)

		st.subheader("LEMMATIZE WORDS")
		text = lemmatize_words(text)
		st.write(text)
		time.sleep(1.5)

		st.subheader("CORRECT SPELLING")
		text = correct_spellings(text)
		text = text.replace("ovid","covid")
		st.write(text)
		time.sleep(1.5)

		st.header("ORIGINAL TEXT")
		st.write(input_text)
		
		st.header("AFTER PREPROCESS")
		st.write(text)


def check_raw_text(raw_text):
    raw_text = preprocess_text(raw_text)
    global current_dir
    current_dir = os.getcwd() + "/"
    print(current_dir)
    transf = vectorizer.transform([raw_text])
    print(transf)
    pred_df = pd.DataFrame(transf.todense(), columns=vectorizer.get_feature_names())[df.columns]
    print(pred_df)
    predict(pred_df)


def initialize_gui():
    #initialize sidebar elements
    global classifier_type
    global text_input_method
    global checkbox
    global checkbox_legend
    classifier_type = st.sidebar.selectbox("WHICH CLASSIFIER WOULD YOU LIKE TO USE?",
                                           ("LOGISTIC REGRESSION","NEURAL NETWORK"))
    checkbox = st.sidebar.checkbox("SHOW PREPROCESS STEPS")
    checkbox_legend = st.sidebar.checkbox("SHOW PREDICTION LEGEND")
    nltk.download('punkt')

st.title("COVID-19 Tweet Sentiment Analysis")
#st.header("What type of upload method would you like to use?")
df = pd.read_csv("csv/headers.csv").drop('Sentiment', axis=1)
#initialize models(Logistic Regression and Neural Network)
LR = pickle.load(open("models/logistic_regression.pk", "rb"))
NN = load_model('models/NN.h5')
vectorizer = pickle.load(open("models/vectorizer.pk", "rb"))
initialize_gui()
input_text = st.text_area("Enter the tweet")
input_text = fix_input_text(input_text)


if st.button("Make prediction"):
	if not input_text:  # if text is empty
		st.warning("Empty Text")
	else:
		st.success("Text processed correctly")
		check_raw_text(input_text)

if checkbox:
    print_preprocess_steps()

if checkbox_legend:
	print_prediction_legend()

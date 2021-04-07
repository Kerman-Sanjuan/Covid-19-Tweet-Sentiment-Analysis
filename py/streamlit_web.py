import streamlit as st
from io import StringIO
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from preproc import preprocess_text
from keras.models import load_model
import nltk

global current_dir
global classifier_type
global text_input_method
global checkbox

def show_word_clouds():
    st.subheader("TRAINING SET WORDCLOUDS")
    images = ['py/wc_neg.png', 'py/wc_pos.png', 'py/wc_neu.png']
    st.image(images, width=500, caption=["NEGATIVE","POSITIVE","NEUTRAL"])


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
		v = np.argmax(y[0])
		if v == 0:
			st.error("Sentiment: NEGATIVE")
		elif v == 1:
			st.warning("Sentiment: NEUTRAL")
		elif v == 2:
			st.success("Sentiment: POSITIVE")        

def check_raw_text(raw_text):
    raw_text = preprocess_text(raw_text)
    global current_dir
    current_dir = os.getcwd() + "/"
    print(current_dir)
    transf = vectorizer.transform([raw_text])
    pred_df = pd.DataFrame(transf.todense(), columns=vectorizer.get_feature_names())[df.columns]
    predict(pred_df)


def process(text):
    if st.button("Make prediction"):
        if not str(text):  # if text is empty
            st.error("Empty Text")
        else:
            st.success("Text processed correctly")
            check_raw_text(text)


def initialize_gui():
    #initialize sidebar elements
    global classifier_type
    global text_input_method
    global checkbox
    text_input_method = st.sidebar.selectbox("UPLOAD METHOD", ("RAW TEXT","EXTERNAL FILE"))
    classifier_type = st.sidebar.selectbox("WHICH CLASSIFIER WOULD YOU LIKE TO USE?",
                                           ("LOGISTIC REGRESSION","NEURAL NETWORK"))
    checkbox = st.sidebar.checkbox("SHOW TRAINING SET WORDCLOUDS")
    nltk.download('punkt')

st.title("COVID-19 Tweet Sentiment Analisys")
#st.header("What type of upload method would you like to use?")
df = pd.read_csv("csv/headers.csv").drop('Sentiment', axis=1)
#initialize models(Logistic Regression and Neural Network)
LR = pickle.load(open("models/logistic_regression.pk", "rb"))
NN = load_model('models/NN.h5')
vectorizer = pickle.load(open("models/vectorizer.pk", "rb"))
initialize_gui()


if text_input_method == 'RAW TEXT':
    process(st.text_area("Enter the tweet"))

else:
    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        process(string_data)

if checkbox:
    show_word_clouds()

import streamlit as st
from io import StringIO
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
global current_dir

def predict(file):
    csv_file = current_dir + "prediction.csv"
    txt_file = current_dir + file.name
    os.system("python3 raw_to_csv.py " + txt_file + " " + csv_file)
    os.remove(txt_file)

def create_txt_file(raw_text):
    file = open("temp.txt","w")
    file.write(raw_text)
    file.close()
    return file

def check_raw_text(raw_text):
    global current_dir
    current_dir = os.getcwd()+"/"
    if st.button("Make prediction"):
        if not str(raw_text):  # if the text is empty
            st.error("Empty Text")
        else:
            st.success("Text processed correctly")
            transf = vectorizer.transform([raw_text])
            pred_df = pd.DataFrame(transf.todense(),columns=vectorizer.get_feature_names())[df.columns]
            y = LR.predict(pred_df)
            # predict(create_txt_file(raw_text))
            if (y[0] == 0):
                st.error("AGAINST COVID TWEET")
            elif (y[0] == 1):
                st.warning("NEUTRAL COVID TWEET")
            else:
                st.success("FAVORABLE COVID TWEET")


df = pd.read_csv("best_attr.csv").drop('Sentiment', axis=1)
LR = pickle.load(open("logistic_regression.pk", "rb"))
vectorizer = pickle.load(open("vectorizer.pk", "rb"))
st.title("Natural Language Processing Web Application")
st.header("What type of upload method would you like to use?")

option = st.selectbox('Upload method',('Raw Text', 'External File')) 


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


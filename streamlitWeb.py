from textblob import TextBlob
import streamlit as st

def predict(value):
	

st.title("Natural Language Processing Web Application")
st.header("What type of upload method would you like to use?	")

option = st.selectbox('Upload method',('Raw Text', 'External File')) 

if option == 'Raw Text':
	raw = st.text_area("Enter the tweet")
	if st.button("Make prediction"):
		predict(raw)
else:
	#upload method
	algo = ""
	if st.button("Make prediction"):
		predict(algo)

st.header("Results")

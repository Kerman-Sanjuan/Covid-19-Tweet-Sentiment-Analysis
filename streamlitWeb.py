import streamlit as st
from io import StringIO
import magic
import os

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
		if not str(raw_text): # if the text is empty
			st.error("Empty Text")
		else:
			st.success("Text processed correctly")
			predict(create_txt_file(raw_text))


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


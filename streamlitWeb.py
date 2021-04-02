import streamlit as st

st.title('TITLE')

dataset_name = st.sidebar.selectbox(
    'Select an upload method',
    ('RAW TEXT', 'TXT FILE')
)
st.write(f"## {dataset_name}")

if dataset_name == "RAW TEXT":
	new_user = st.text_input("TWEET")
	if st.button('Make prediction'):
		st.write("Input: " + new_user)

else:	
	uploaded_file = st.file_uploader("Choose a file")
	if uploaded_file is not None:
		if st.button('Make prediction'):
			st.write("Input: " + new_user)
	else:
		st.write("No file has been uploaded...")

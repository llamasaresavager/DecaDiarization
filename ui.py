import streamlit as st
import requests
import os
import platform
import time

# Constants
FASTAPI_URL = 'http://127.0.0.1:8000'  # Update this as needed

# Set the default downloads path based on the operating system
if platform.system() == "Windows":
    default_path = os.path.join(os.path.expanduser("~"), "Downloads")
elif platform.system() == "Darwin":
    default_path = os.path.join(os.path.expanduser("~"), "Downloads")
else:  # Assuming Linux
    default_path = os.path.join(os.path.expanduser("~"), "Downloads")

# Streamlit UI
st.title("Audio Diaritization and Transcription Service")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")
operation = st.radio("Select operation", ('Diaritize', 'Transcribe'))
output_filename = st.text_input('Enter output filename', value='output.json')
destination = st.text_input('Enter destination directory', value=default_path)
submit = st.button('Start')

if uploaded_file is not None and destination:
    if not os.path.isdir(destination):
        st.error('Destination directory does not exist')
    elif submit:
        file = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'audio/wav')}
        endpoint = "/diarize" if operation == 'Diaritize' else "/transcribe"

        with st.spinner('Processing...'):
            response = requests.post(f'{FASTAPI_URL}{endpoint}', files=file)

        if response.status_code == 200:
            filename = os.path.join(destination, output_filename)
            with open(filename, 'w') as f:
                f.write(response.text)
            st.success(f'Result saved to {filename}')
        else:
            st.error(f'Error: {response.text}')

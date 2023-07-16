import streamlit as st
import requests
import base64
import time

# Constants
FASTAPI_URL = 'http://127.0.0.1:8000'  # Update this as needed

# Function to check if the FastAPI server is running
def is_server_running(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Wait for the FastAPI server to start
server_message = st.empty()
server_message.info("Waiting for the FastAPI server to start. This should only take a moment...")
while not is_server_running(f'{FASTAPI_URL}/status'):
    time.sleep(1)
server_message.empty()
st.success("FastAPI server is running!")

# Streamlit UI
st.title("Audio Diaritization and Transcription Service")

huggingface_api_key = st.text_input('Enter your Hugging Face API key needed for some models (Optional)', value='')
# A small info text box
st.info("By default, this program uses the model pyannote/speaker-diarization, which requires you to accept the conditions prior to use. See more here: https://huggingface.co/pyannote/speaker-diarization")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")
operation = st.radio("Select operation", ('Diaritize', 'Transcribe'))
default_filename = 'output_diarize.json' if operation == 'Diaritize' else 'output_transcribe.json'
output_filename = st.text_input('Enter output filename', value=default_filename)
submit = st.button('Start Processing')

# Function to create a download link
def create_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="display: inline-block; padding: .375rem .75rem; font-size: 1rem; line-height: 1.5; text-align: center; white-space: nowrap; vertical-align: middle; border: 1px solid transparent; border-radius: .25rem; color: #fff; background-color: #007bff; text-decoration: none;">Download {filename}</a>'

# Function to send the file to the server
def send_file_to_server(file, endpoint, huggingface_api_key=None):
    try:
        data = {'huggingface_api_key': huggingface_api_key} if huggingface_api_key else {}
        response = requests.post(f'{FASTAPI_URL}{endpoint}', files=file, data=data)
        response.raise_for_status()  # raise an exception in case of error
    except requests.exceptions.RequestException as err:
        st.error(f'Error: {str(err)}')
        return None
    return response

if uploaded_file is not None and submit:
    upload_message = st.empty()
    upload_message.info("File uploaded successfully!")
    file = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'audio/wav')}
    endpoint = "/diarize" if operation == 'Diaritize' else "/transcribe"
    response = send_file_to_server(file, endpoint, huggingface_api_key)

    if response is not None:
        upload_message.empty()
        st.success('Processing completed successfully.')
        # The response content will be the output file content
        output_content = response.text
        # Create a download link with the custom filename for the output file
        download_link = create_download_link(output_content, output_filename)
        st.markdown(download_link, unsafe_allow_html=True)

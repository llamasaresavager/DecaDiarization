# Audio Diarization and Transcription with FastAPI

This repository contains a Python project that performs audio diarization and transcription on a streaming WAV file using FastAPI and a Streamlit user interface.

![Streamlit UI](./images/ui.png)
## Project Structure

The project consists of the following files:

- `Diarization.py`: Contains the diarization logic for processing an audio stream and extracting speaker segments.
- `Transcription.py`: Contains the transcription logic for converting speaker segments into text.
- `app.py`: Defines the FastAPI application and endpoints for handling the audio diarization and transcription.
- `streamlit_app.py`: Defines the Streamlit user interface for file upload, request handling, and result display.
- `main.py`: The main Python script to run both the FastAPI server and the Streamlit UI.
- `requirements.txt`: Lists the required Python packages for running the project.
- `README.md`: Provides information about the project and instructions for running it.

## Installation

1. Create a new Conda environment with Python 3.7:

    ```bash
    conda create -n myenv python=3.7
    ```

2. Activate the Conda environment:

    ```bash
    conda activate myenv
    ```

3. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/llamasaresavager/DecaDiarization.git
    ```

4. Navigate to the project directory:

    ```bash
    cd DecaDiarization
    ```

5. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. To start the FastAPI server and the Streamlit UI, run the following command in your terminal:

   ```bash
   python main.py
   ```

This will launch the FastAPI server and the Streamlit UI. You can now use the UI to upload a WAV file, select a task (diarization or transcription), and select a destination for the output file. The selected task will be performed on the uploaded file, and the result will be displayed in the Streamlit app and saved to the chosen destination.

Please ensure that you have a valid WAV file for diarization and adjust the request accordingly.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
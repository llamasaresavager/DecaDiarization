# Audio Diarization and Transcription with FastAPI

This repository contains a Python project that performs audio diarization and transcription on a streaming WAV file using FastAPI.

## Project Structure

The project consists of the following files:

- `Diarization.py`: Contains the diarization logic for processing an audio stream and extracting speaker segments.
- `Transcription.py`: Contains the transcription logic for converting speaker segments into text.
- `app.py`: Defines the FastAPI application and endpoints for handling the audio diarization and transcription.
- `requirements.txt`: Lists the required Python packages for running the project.
- `README.md`: Provides information about the project and instructions for running it.

## Installation

1. Create a new Conda environment with Python 3.7:

    ```conda create -n myenv python=3.7```

2. Activate the Conda environment:

    ```conda activate myenv```

3. Clone the repository to your local machine:

    ```git clone https://github.com/llamasaresavager/DecaDiarization.git```

4. Navigate to the project directory:

    ```cd DecaDiarization```

5. Install the required Python packages:

    ```pip install -r requirements.txt```

## Usage

1. To start the FastAPI server, run the following command in your terminal:

    ```bash uvicorn app:app --reload ```

2. Send a POST request to `http://localhost:8000/diarize` with the WAV file as the request body. The server will perform audio diarization and return the result as a JSON object.

3. Send a POST request to `http://localhost:8000/transcribe` with the WAV file and diarization result as the request body. The server will perform audio transcription and return the result as a JSON object.

Please ensure that you have a valid WAV file for diarization and adjust the request accordingly.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

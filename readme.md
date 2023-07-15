# Audio Diarization with FastAPI

This repository contains a Python project that performs audio diarization on a streaming WAV file using FastAPI.

## Project Structure

The project consists of the following files:

- `Diarization.py`: Contains the diarization logic for processing an audio stream and extracting speaker segments.
- `app.py`: Defines the FastAPI application and endpoint for handling the audio diarization.
- `requirements.txt`: Lists the required Python packages for running the project.
- `README.md`: Provides information about the project and instructions for running it.

## Installation

1. Create a new Conda environment with Python 3.7:

    ```conda create -n myenv python=3.7```


2. Activate the Conda environment:

    ```conda activate myenv```


3. Install the required Python packages:

    ```pip install -r requirements.txt```

4. Clone the repository to your local machine:

    ```git clone https://github.com/llamasaresavager/DecaDiarization.git```

5. Navigate to the project directory:

    ```cd DecaDiarization```

6. Install the required Python packages:

    ```pip install -r requirements.txt```


## Usage

1. To start the FastAPI server, run the following command in your terminal:

    ```bash uvicorn app:app --reload ```


2. Send a POST request to `http://localhost:8000/diarize` with the WAV file as the request body. The server will perform audio diarization and return the result as a JSON object.

Please ensure that you have a valid WAV file for diarization and adjust the request accordingly.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

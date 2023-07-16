from noisereduce import reduce_noise
from soundfile import read, _error_check
from librosa.effects import trim
from librosa.util import normalize
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch import cuda, device
import io
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)


class Transcriber:
    CHUNK_SIZE_SEC = 5
    MODEL_NAME = "openai/whisper-large-v2"

    def __init__(self):
        """
        Initialize the Transcriber with the appropriate model and processor.
        """
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(self.MODEL_NAME)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.MODEL_NAME).to(self.device)

    def _preprocess_audio(self, audio_stream: bytes) -> np.array:
        """
        Preprocess audio data by reducing noise and normalizing volume.

        Parameters:
            audio_stream (bytes): The audio data to preprocess.

        Returns:
            np.array: The preprocessed audio data.
        """
        try:
            raw_data, samplerate = read(io.BytesIO(audio_stream))
            data = reduce_noise(y=raw_data, sr=samplerate)
            data, _ = trim(data) 
            data = normalize(data)
            return data, samplerate
        except _error_check as e:
            logging.error(f"Error reading audio file: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def _transcribe_chunk(self, chunk: np.array, samplerate: int, speaker: str, audio_id: str, timestamp: str) -> list:
        """
        Transcribe an audio chunk using the Whisper ASR model.

        Parameters:
            chunk (np.array): The audio chunk to transcribe.
            samplerate (int): The samplerate of the audio chunk.
            speaker (str): The speaker of the audio chunk.
            audio_id (str): An ID for the audio chunk.
            timestamp (str): A timestamp for the audio chunk.

        Returns:
            list: The transcription results.
        """
        input_features = self.processor(chunk, sampling_rate=samplerate, return_tensors="pt").input_features.to(self.device)
        predicted_ids = self.model.generate(input_features)
        transcriptions = self.processor.batch_decode(predicted_ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        transcript_chunk = [{"audio_id": audio_id, "timestamp": timestamp, "Speaker": speaker, "transcription": t} for t in transcriptions]
        return transcript_chunk

    def _process_diarization(self, data, samplerate, diarization_result, audio_id, timestamp):
        transcript_json = []
        for segment in diarization_result:
            start_sample = int(segment["sent_start"] * samplerate)
            end_sample = int(segment["sent_end"] * samplerate)
            chunk = data[start_sample:end_sample]
            transcript_chunk = self._transcribe_chunk(chunk, samplerate, segment["Speaker"], audio_id, timestamp)
            transcript_json.extend(transcript_chunk)
        return transcript_json

    def _process_without_diarization(self, data, samplerate, chunk_size_sec, audio_id, timestamp):
        transcript_json = []
        total_samples = len(data)
        chunk_size_samples = chunk_size_sec * samplerate
        for start_sample in range(0, total_samples, chunk_size_samples):
            end_sample = min(start_sample + chunk_size_samples, total_samples)
            chunk = data[start_sample:end_sample]
            transcript_chunk = self._transcribe_chunk(chunk, samplerate, "Unknown", audio_id, timestamp)
            transcript_json.extend(transcript_chunk)
        return transcript_json

    def transcribe_audio_file(self, audio_stream: bytes, diarization_result: list, audio_id: str=None, timestamp: str=None, do_diarize: bool=True, chunk_size_sec: int=CHUNK_SIZE_SEC) -> list:
        data, samplerate = self._preprocess_audio(audio_stream)
        if do_diarize:
            return self._process_diarization(data, samplerate, diarization_result, audio_id, timestamp)
        else:
            return self._process_without_diarization(data, samplerate, chunk_size_sec, audio_id, timestamp)

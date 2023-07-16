from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import io
import torch
import librosa
import noisereduce as nr

def transcribe_audio_file(audio_stream, diarization_result, audio_id=None, timestamp=None, do_diarize=True, chunk_size_sec=5):
    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read the audio stream with soundfile
    raw_data, samplerate = sf.read(io.BytesIO(audio_stream))

    # Reduce noise
    data = nr.reduce_noise(y=raw_data, sr=samplerate)

    # Minor preprocessing with Librosa
    data, _ = librosa.effects.trim(data)  # Trim leading and trailing silence
    data = librosa.util.normalize(data)  # Normalize amplitude to range [-1, 1]

    # Load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)

    # Prepare the transcript JSON object
    transcript_json = []

    if do_diarize:
        # Process the audio in chunks based on diarization result
        for segment in diarization_result:
            start_sample = int(segment["sent_start"] * samplerate)
            end_sample = int(segment["sent_end"] * samplerate)
            chunk = data[start_sample:end_sample]

            # Process this chunk
            transcript_chunk = transcribe_chunk(processor, model, device, chunk, samplerate, segment["Speaker"], audio_id, timestamp)
            transcript_json.extend(transcript_chunk)
    else:
        # Without diarization, chunk the audio by fixed time segments
        total_samples = len(data)
        chunk_size_samples = chunk_size_sec * samplerate
        for start_sample in range(0, total_samples, chunk_size_samples):
            end_sample = min(start_sample + chunk_size_samples, total_samples)
            chunk = data[start_sample:end_sample]

            # Process this chunk
            transcript_chunk = transcribe_chunk(processor, model, device, chunk, samplerate, "Unknown", audio_id, timestamp)
            transcript_json.extend(transcript_chunk)

    return transcript_json


def transcribe_chunk(processor, model, device, chunk, samplerate, speaker, audio_id, timestamp):
    # Transcribe the audio chunk directly without involving dataset splits
    input_features = processor(chunk, sampling_rate=samplerate, return_tensors="pt").input_features.to(device)

    # Generate token ids
    predicted_ids = model.generate(input_features)

    # Decode token ids to text
    transcriptions = processor.batch_decode(predicted_ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Append the transcriptions for this chunk
    transcript_chunk = [{
        "audio_id": audio_id,        # ID or name of the audio stream (optional)
        "timestamp": timestamp,      # Timestamp of the audio stream (optional)
        "Speaker": speaker,          # Speaker for this segment
        "transcription": t           # Transcribed text for the segment
    } for t in transcriptions]

    return transcript_chunk

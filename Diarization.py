import pandas as pd
from pyannote.audio import Pipeline, Audio
import tempfile
import soundfile as sf

def diarize(stream):
    # Save the audio_data to a temporary .wav file
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".wav")
    sf.write(temp_file.name, audio_data, 44100)


    # Load the pre-trained diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="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"
    )

    # Perform diarization on the temporary audio file
    diarization = pipeline(temp_file.name)

    # Perform diarization on the input audio stream
    diarization = pipeline(x)

    # Convert the diarization result to a JSON object
    diar_json = diarization.for_json()

    # Convert the JSON object to a DataFrame
    diar_df = pd.DataFrame.from_dict(diar_json["content"])

    # Extract the start and end times for each segment
    diar_df["start"] = [diar_df["segment"][i]["start"] for i in range(len(diar_df["segment"]))]
    diar_df["end"] = [diar_df["segment"][i]["end"] for i in range(len(diar_df["segment"]))]
    diar_df = diar_df[["label", "start", "end"]]

    # Define how to aggregate and group the segments
    d = {"label": "first", "start": "min", "end": "max"}  # How to aggregate
    s = diar_df.label.ne(diar_df.label.shift(1)).cumsum().rename(None)  # How to group
    diar_df = diar_df.groupby(s).agg(d)

    # Reset the index and rename the columns
    diar_df.reset_index(inplace=True)
    diar_df = diar_df[["label", "start", "end"]]
    diar_df.columns = ["Speaker", "sent_start", "sent_end"]

    return diar_df

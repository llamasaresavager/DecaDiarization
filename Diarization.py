import os
import traceback
import pandas as pd
from pyannote.audio import Pipeline, Audio
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def diarize(stream):
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            temp_file.write(stream)
            
            huffingface_auth_token = os.environ.get("HUFFINGFACE_AUTH_TOKEN")
            if not huffingface_auth_token:
                raise Exception("Environment variable HUFFINGFACE_AUTH_TOKEN is not set")
            
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=huffingface_auth_token
            )

            diarization = pipeline(temp_file.name)
            diar_json = diarization.for_json()
            return json_to_df(diar_json)

    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        raise Exception("".join(tb_str))

def json_to_df(diar_json):
    segments = diar_json["content"]
    df = pd.DataFrame(segments)

    df['start'] = df['segment'].apply(lambda x: x['start'])
    df['end'] = df['segment'].apply(lambda x: x['end'])

    df.drop(columns=['segment'], inplace=True)

    df = df.groupby((df['label'] != df['label'].shift(1)).cumsum()).agg({
        'label': 'first',
        'start': 'min',
        'end': 'max',
    })

    df.reset_index(drop=True, inplace=True)

    df.rename(columns={'label': 'Speaker', 'start': 'sent_start', 'end': 'sent_end'}, inplace=True)

    return df

import pandas as pd
import shutil
from glob import glob
from pyannote.audio import Pipeline


##Unneccesery for steaming
file = 'output-trim.wav'
##Unneccesery for steaming
def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data
            

def diarize(file):

    ##Unneccesery for steaming
    shutil.copy(file, 'audio_in/')

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")

    diarization = pipeline(file)
    diar_json = diarization.for_json()
    diar_df = pd.DataFrame.from_dict(diar_json['content'])

    diar_df['start'] = [diar_df['segment'][i]['start'] for i in range(len(diar_df['segment']))]
    diar_df['end'] = [diar_df['segment'][i]['end'] for i in range(len(diar_df['segment']))]
    diar_df = diar_df[['label', 'start', 'end']]

    d = {'label': 'first', 'start': 'min', 'end': 'max'}   # How to aggregate
    s = diar_df.label.ne(diar_df.label.shift(1)).cumsum().rename(None) # How to group
    print(diar_df)
    diar_df = diar_df.groupby(s).agg(d)


    
    diar_df.reset_index(inplace=True)
    diar_df = diar_df[['label', 'start', 'end']]
    diar_df.columns = ['Speaker', 'sent_start', 'sent_end']


    return diar_df

diarize(file=file)
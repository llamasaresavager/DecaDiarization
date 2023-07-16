from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import soundfile as sf
from io import BytesIO
import numpy as np

# Import the function from Diarization.py
from Diarization import diarize

class Segment(BaseModel):
    Speaker: str
    sent_start: float
    sent_end: float

app = FastAPI()

@app.post("/diarize", response_model=List[Segment])
async def diarization(file: UploadFile = File(...)):
    try:
        # Ensure the file type is correct
        if file.content_type != "audio/wav":
            raise HTTPException(status_code=400, detail="File must be a .wav file")

        # Load the audio data
        audio_data, sample_rate = sf.read(BytesIO(await file.read()))

        # Perform diarization
        diar_df = diarize(audio_data)

        # Convert the DataFrame to a list of dicts
        diar_list = diar_df.to_dict(orient='records')

        return JSONResponse(content=diar_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

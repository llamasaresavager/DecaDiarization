from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import traceback
from Diarization import diarize
from Transcription import transcribe_audio_file

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

        # Perform diarization
        diar_df = diarize(await file.read())

        # Convert the DataFrame to a list of dicts
        diar_list = diar_df.to_dict(orient='records')

        return JSONResponse(content=diar_list)

    except Exception as e:
        # Capture the full exception traceback and raise as HTTPException
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        raise HTTPException(status_code=500, detail="".join(tb_str))


@app.post("/transcribe", response_model=List[Segment])
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Ensure the file type is correct
        if file.content_type != "audio/wav":
            raise HTTPException(status_code=400, detail="File must be a .wav file")

        # Perform transcription
        transcript = transcribe_audio_file(await file.read())

        return JSONResponse(content=transcript)

    except Exception as e:
        # Capture the full exception traceback and raise as HTTPException
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        raise HTTPException(status_code=500, detail="".join(tb_str))

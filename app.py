import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.params import Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
from Diarization import diarize
from Transcription import Transcriber
import traceback

class Segment(BaseModel):
    Speaker: str
    sent_start: float
    sent_end: float

class Transcript(BaseModel):
    audio_id: str
    timestamp: float
    transcription: str

app = FastAPI()
transcriber = Transcriber()

def validate_file_type(file: UploadFile):
    if file.filename.split(".")[-1] not in ["wav"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

@app.post("/diarize", response_model=List[Segment])
async def diarization(file: UploadFile = File(...)):
    try:
        validate_file_type(file)
        diar_df = diarize(await file.read())
        diar_list = diar_df.to_dict(orient='records')
        return JSONResponse(content=diar_list)

    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        raise HTTPException(status_code=500, detail="".join(tb_str))

@app.post("/transcribe", response_model=List[Transcript])
async def transcribe_audio(
    file: UploadFile = File(...),
    do_diarize: Optional[bool] = True,
    huggingface_api_key: Optional[str] = Form(None),  # Include the Hugging Face API key as an optional form parameter
):
    try:
        validate_file_type(file)

        if huggingface_api_key:  # If the API key is provided, set it as an environment variable
            os.environ['HUGGINGFACE_API_KEY'] = huggingface_api_key

        file_contents = await file.read()

        diar_df = diarize(file_contents) if do_diarize else None

        transcript = transcriber.transcribe_audio_file(
            file_contents,
            diar_df.to_dict(orient='records') if diar_df is not None and not diar_df.empty else None,
            do_diarize=do_diarize
        )

        return JSONResponse(content=transcript)

    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        raise HTTPException(status_code=500, detail="".join(tb_str))
    
@app.get("/status")
async def read_status():
    return {"status": "OK"}
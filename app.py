from fastapi import FastAPI, UploadFile, HTTPException
from tempfile import NamedTemporaryFile
from typing import List
from Diarization import diarize

app = FastAPI()

@app.post("/diarize")
async def diarize_audio(files: List[UploadFile]):
    if len(files) != 1:
        raise HTTPException(status_code=400, detail="Exactly one file should be provided.")

    file = files[0]
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    
    with file.file as fx:
        fx.seek(0)
        # x= fx.read(10)
        # diarization_result = x
        # Perform diarization on the temporary file stream
        diarization_result = diarize(stream=fx)

    return diarization_result

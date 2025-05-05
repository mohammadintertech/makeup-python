from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
from lipstick_processor import apply_lipstick
from eyeliner_processor import apply_eyeliner
from blusher_processor import apply_blusher
from foundation_processor import apply_foundation
from eyecolor_processor import apply_eyecolor
from eye_shadow_processor import apply_eyeshadow
from typing import Optional

app = FastAPI()

@app.post("/apply")
async def apply_makeup(
    file: UploadFile = File(...),

    # Enable flags
    enable_lipstick: bool = Form(False),
    enable_eyeliner: bool = Form(False),
    enable_blusher: bool = Form(False),
    enable_foundation: bool = Form(False),
    enable_eyecolor: bool = Form(False),
    enable_eyeshadow: bool = Form(False),

    # Lipstick
    lipstick_r: Optional[int] = Form(None),
    lipstick_g: Optional[int] = Form(None),
    lipstick_b: Optional[int] = Form(None),
    lipstick_intensity: Optional[float] = Form(None),
    lipstick_edge: Optional[int] = Form(None),

    # Eyeliner
    eyeliner_r: Optional[int] = Form(None),
    eyeliner_g: Optional[int] = Form(None),
    eyeliner_b: Optional[int] = Form(None),

    # Blusher
    blusher_r: Optional[int] = Form(None),
    blusher_g: Optional[int] = Form(None),
    blusher_b: Optional[int] = Form(None),
    blusher_intensity: Optional[float] = Form(None),

    # Foundation
    foundation_r: Optional[int] = Form(None),
    foundation_g: Optional[int] = Form(None),
    foundation_b: Optional[int] = Form(None),
    foundation_intensity: Optional[float] = Form(None),

    # Eye color
    eyecolor_r: Optional[int] = Form(None),
    eyecolor_g: Optional[int] = Form(None),
    eyecolor_b: Optional[int] = Form(None),

    # Eyeshadow
    eyeshadow_r: Optional[int] = Form(None),
    eyeshadow_g: Optional[int] = Form(None),
    eyeshadow_b: Optional[int] = Form(None),
    eyeshadow_intensity: Optional[float] = Form(None)
):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        processed = image.copy()

        if enable_foundation and all(v is not None for v in [foundation_r, foundation_g, foundation_b, foundation_intensity]):
            processed = apply_foundation(processed, [foundation_r, foundation_g, foundation_b], foundation_intensity)

        if enable_blusher and all(v is not None for v in [blusher_r, blusher_g, blusher_b, blusher_intensity]):
            processed = apply_blusher(processed,[ blusher_r, blusher_g, blusher_b], blusher_intensity)

        if enable_lipstick and all(v is not None for v in [lipstick_r, lipstick_g, lipstick_b, lipstick_intensity, lipstick_edge]):
            processed = apply_lipstick(processed, [lipstick_r, lipstick_g, lipstick_b], lipstick_intensity, lipstick_edge)

        if enable_eyeliner and all(v is not None for v in [eyeliner_r, eyeliner_g, eyeliner_b]):
            processed = apply_eyeliner(processed, [eyeliner_r, eyeliner_g, eyeliner_b])

        if enable_eyecolor and all(v is not None for v in [eyecolor_r, eyecolor_g, eyecolor_b]):
            processed = apply_eyecolor(processed, eyecolor_r, eyecolor_g, eyecolor_b)

        if enable_eyeshadow and all(v is not None for v in [eyeshadow_r, eyeshadow_g, eyeshadow_b, eyeshadow_intensity]):
            processed = apply_eyeshadow(processed,[ eyeshadow_r, eyeshadow_g, eyeshadow_b], eyeshadow_intensity)

        _, img_encoded = cv2.imencode(".png", processed)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

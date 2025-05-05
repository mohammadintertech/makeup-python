from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, Body
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
from pydantic import BaseModel

app = FastAPI()



# Lipstick API Endpoint
@app.get("/")
async def process_image(
  
):
  

    return 'Helo world!'




class MakeupSettings(BaseModel):
    # Enable flags for each effect
    enable_lipstick: Optional[bool] = False
    enable_eyeliner: Optional[bool] = False
    enable_blusher: Optional[bool] = False
    enable_foundation: Optional[bool] = False
    enable_eyecolor: Optional[bool] = False
    enable_eyeshadow: Optional[bool] = False
    
    # Lipstick parameters
    lipstick_r: Optional[int] = None
    lipstick_g: Optional[int] = None
    lipstick_b: Optional[int] = None
    lipstick_intensity: Optional[float] = None
    lipstick_edge: Optional[int] = None
    
    # Eyeliner parameters
    eyeliner_r: Optional[int] = None
    eyeliner_g: Optional[int] = None
    eyeliner_b: Optional[int] = None
    
    # Blusher parameters
    blusher_r: Optional[int] = None
    blusher_g: Optional[int] = None
    blusher_b: Optional[int] = None
    blusher_intensity: Optional[float] = None
    
    # Foundation parameters
    foundation_r: Optional[int] = None
    foundation_g: Optional[int] = None
    foundation_b: Optional[int] = None
    foundation_intensity: Optional[float] = None
    
    # Eye color parameters
    eyecolor_r: Optional[int] = None
    eyecolor_g: Optional[int] = None
    eyecolor_b: Optional[int] = None
    
    # Eyeshadow parameters
    eyeshadow_r: Optional[int] = None
    eyeshadow_g: Optional[int] = None
    eyeshadow_b: Optional[int] = None
    eyeshadow_intensity: Optional[float] = None

@app.post("/apply")
async def apply_makeup(
    file: UploadFile = File(...),
    settings: MakeupSettings = Body(...)
):
    try:
        # Read and decode the image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        processed = image.copy()
        
        # Apply foundation if enabled and parameters provided
        if (settings.enable_foundation and 
            settings.foundation_r is not None and 
            settings.foundation_g is not None and 
            settings.foundation_b is not None and 
            settings.foundation_intensity is not None):
            processed = apply_foundation(
                processed, 
                [settings.foundation_r, settings.foundation_g, settings.foundation_b], 
                settings.foundation_intensity
            )
        
        # Apply blusher if enabled and parameters provided
        if (settings.enable_blusher and 
            settings.blusher_r is not None and 
            settings.blusher_g is not None and 
            settings.blusher_b is not None and 
            settings.blusher_intensity is not None):
            processed = apply_blusher(
                processed, 
                settings.blusher_r, 
                settings.blusher_g, 
                settings.blusher_b, 
                settings.blusher_intensity
            )
        
        # Apply lipstick if enabled and parameters provided
        if (settings.enable_lipstick and 
            settings.lipstick_r is not None and 
            settings.lipstick_g is not None and 
            settings.lipstick_b is not None and 
            settings.lipstick_intensity is not None and 
            settings.lipstick_edge is not None):
            processed = apply_lipstick(
                processed, 
                [settings.lipstick_b, settings.lipstick_g, settings.lipstick_r], 
                settings.lipstick_intensity, 
                settings.lipstick_edge
            )
        
        # Apply eyeliner if enabled and parameters provided
        if (settings.enable_eyeliner and 
            settings.eyeliner_r is not None and 
            settings.eyeliner_g is not None and 
            settings.eyeliner_b is not None):
            processed = apply_eyeliner(
                processed, 
                settings.eyeliner_r, 
                settings.eyeliner_g, 
                settings.eyeliner_b
            )
        
        # Apply eye color if enabled and parameters provided
        if (settings.enable_eyecolor and 
            settings.eyecolor_r is not None and 
            settings.eyecolor_g is not None and 
            settings.eyecolor_b is not None):
            processed = apply_eyecolor(
                processed, 
                settings.eyecolor_r, 
                settings.eyecolor_g, 
                settings.eyecolor_b
            )
        
        # Apply eyeshadow if enabled and parameters provided
        if (settings.enable_eyeshadow and 
            settings.eyeshadow_r is not None and 
            settings.eyeshadow_g is not None and 
            settings.eyeshadow_b is not None and 
            settings.eyeshadow_intensity is not None):
            processed = apply_eyeshadow(
                processed, 
                settings.eyeshadow_r, 
                settings.eyeshadow_g, 
                settings.eyeshadow_b, 
                settings.eyeshadow_intensity
            )
        
        # Encode and return the processed image
        _, img_encoded = cv2.imencode(".png", processed)
        return StreamingResponse(
            io.BytesIO(img_encoded.tobytes()), 
            media_type="image/png"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



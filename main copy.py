from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
from fastapi import Body

app = FastAPI()

# Lipstick API Endpoint
@app.post("/apply-lipstick")
async def process_image(
    file: UploadFile = File(...),
    r: int = Form(255),
    g: int = Form(0),
    b: int = Form(0),
    intensity: float = Form(0.8),
    edge: int = Form(10)
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        processed = apply_lipstick(image, [b, g, r], intensity, edge)
    except Exception as e:
        return {"error": str(e)}

    _, img_encoded = cv2.imencode(".png", processed)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")




# Eyeliner API Endpoint
@app.post("/apply-eyeliner")
async def apply_eyeliner_api(
    file: UploadFile = File(...),  # Image file
    r: int = Form(...),  # Red value
    g: int = Form(...),  # Green value
    b: int = Form(...),  # Blue value
):
    try:
        # Read the file content as bytes
        file_content = await file.read()

        # Call the function to apply the eyeliner with RGB values
        processed_image = apply_eyeliner(file_content, r, g, b)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="No face detected")

        # Return the processed image as response
        return Response(content=processed_image, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@app.post("/apply-blusher")
async def apply_blusher_endpoint(
    file: UploadFile = File(...),
    r: int = Form(...),  # Red color component
    g: int = Form(...),  # Green color component
    b: int = Form(...),  # Blue color component
    intensity: float = Form(...),  # Intensity factor
):
    try:
        # Read the file content as bytes
        file_content = await file.read()

        # Apply blusher
        processed_image = apply_blusher(file_content, r, g, b, intensity)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="No face detected")

        # Convert the processed image to bytes and return
        _, buffer = cv2.imencode('.jpg', processed_image)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply-foundation")
async def apply_foundation_api(
    file: UploadFile = File(...),
    r: int = Form(...),
    g: int = Form(...),
    b: int = Form(...),
    intensity: float = Form(...)
):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result = apply_foundation(image, [r, g, b], intensity)

        if result is None:
            raise HTTPException(status_code=400, detail="No face detected")

        _, buffer = cv2.imencode(".jpg", result)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply-eyecolor")
async def apply_eyecolor_endpoint(
    file: UploadFile = File(...),
    r: int = Form(...),  # Red color component
    g: int = Form(...),  # Green color component
    b: int = Form(...),  # Blue color component
):
    try:
        # Read the file content as bytes
        file_content = await file.read()

        # Convert bytes to image
        img_array = np.frombuffer(file_content, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Apply eye color
        processed_image = apply_eyecolor(image, r, g, b)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="No face detected")

        # Convert the processed image to bytes and return
        _, buffer = cv2.imencode('.jpg', processed_image)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply-eyeshadow")
async def apply_eyeshadow_endpoint(
    file: UploadFile = File(...),
    r: int = Form(...),  # Red color component
    g: int = Form(...),  # Green color component
    b: int = Form(...),  # Blue color component
    intensity: float = Form(...),  # Transparency factor
):
    try:
        file_content = await file.read()

        # Apply eyeshadow effect
        processed_image = apply_eyeshadow(file_content, r, g, b, intensity)

        # Convert the processed image to bytes
        _, buffer = cv2.imencode('.jpg', processed_image)

        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




class ColorParams(BaseModel):
    r: int
    g: int
    b: int
    intensity: Optional[float] = 1.0
    edge: Optional[int] = 10  # Only used by lipstick

class ApplyMakeupRequest(BaseModel):
    lipstickEnable: Optional[bool] = False
    lipstick: Optional[ColorParams] = None

    eyelinerEnable: Optional[bool] = False
    eyeliner: Optional[ColorParams] = None

    blusherEnable: Optional[bool] = False
    blusher: Optional[ColorParams] = None

    foundationEnable: Optional[bool] = False
    foundation: Optional[ColorParams] = None

    eyecolorEnable: Optional[bool] = False
    eyecolor: Optional[ColorParams] = None

    eyeshadowEnable: Optional[bool] = False
    eyeshadow: Optional[ColorParams] = None

# Combined Endpoint
@app.post("/apply")
async def apply_makeup(
    file: UploadFile = File(...),
    params: ApplyMakeupRequest = Body(...)
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        # Lipstick
        if params.lipstickEnable and params.lipstick:
            image = apply_lipstick(
                image,
                [params.lipstick.b, params.lipstick.g, params.lipstick.r],
                params.lipstick.intensity,
                params.lipstick.edge or 10
            )

        # Eyeliner
        if params.eyelinerEnable and params.eyeliner:
            eyeliner_buffer = cv2.imencode(".jpg", image)[1].tobytes()
            image = apply_eyeliner(
                eyeliner_buffer,
                params.eyeliner.r,
                params.eyeliner.g,
                params.eyeliner.b
            )

        # Blusher
        if params.blusherEnable and params.blusher:
            blusher_buffer = cv2.imencode(".jpg", image)[1].tobytes()
            image = apply_blusher(
                blusher_buffer,
                params.blusher.r,
                params.blusher.g,
                params.blusher.b,
                params.blusher.intensity
            )

        # Foundation
        if params.foundationEnable and params.foundation:
            image = apply_foundation(
                image,
                [params.foundation.r, params.foundation.g, params.foundation.b],
                params.foundation.intensity
            )

        # Eye color
        if params.eyecolorEnable and params.eyecolor:
            image = apply_eyecolor(
                image,
                params.eyecolor.r,
                params.eyecolor.g,
                params.eyecolor.b
            )

        # Eye shadow
        if params.eyeshadowEnable and params.eyeshadow:
            shadow_buffer = cv2.imencode(".jpg", image)[1].tobytes()
            image = apply_eyeshadow(
                shadow_buffer,
                params.eyeshadow.r,
                params.eyeshadow.g,
                params.eyeshadow.b,
                params.eyeshadow.intensity
            )

        # Encode final image
        _, final_image = cv2.imencode(".jpg", image)
        return Response(content=final_image.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

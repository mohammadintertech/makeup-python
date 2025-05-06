import firebase_functions
from firebase_functions import https_fn
from firebase_admin import initialize_app
import base64
import json

# Initialize Firebase
initialize_app()

@https_fn.on_request(
    cors=True,
    memory=firebase_functions.options.MemoryOption.GB_2,
    timeout_sec=300  # Increased timeout for image processing
)
def apply_makeup(req: https_fn.Request) -> https_fn.Response:
    # Handle CORS preflight
    if req.method == "OPTIONS":
        return https_fn.Response(
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            status=204
        )

    if req.method != "POST":
        return https_fn.Response("Method Not Allowed", status=405)

    try:
        # Import heavy dependencies only when needed
        import numpy as np
        import cv2
        from io import BytesIO
        
        # Import your processor functions
        from lipstick_processor import apply_lipstick
        from eyeliner_processor import apply_eyeliner
        from blusher_processor import apply_blusher
        from foundation_processor import apply_foundation
        from eyecolor_processor import apply_eyecolor
        from eye_shadow_processor import apply_eyeshadow

        # Get JSON data
        data = req.get_json()
        if not data or "image" not in data:
            return https_fn.Response("Missing image data", status=400)

        # Decode base64 image
        image_bytes = base64.b64decode(data["image"])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return https_fn.Response("Invalid image", status=400)

        processed = image.copy()

        # Process each makeup component
        if data.get("enable_foundation"):
            foundation_color = [
                data.get("foundation_r", 0),
                data.get("foundation_g", 0),
                data.get("foundation_b", 0)
            ]
            intensity = float(data.get("foundation_intensity", 0.5))
            processed = apply_foundation(processed, foundation_color, intensity)

        if data.get("enable_blusher"):
            blusher_color = [
                data.get("blusher_r", 0),
                data.get("blusher_g", 0),
                data.get("blusher_b", 0)
            ]
            intensity = float(data.get("blusher_intensity", 0.5))
            processed = apply_blusher(processed, blusher_color, intensity)

        # Add similar blocks for other makeup types...

        # Encode the processed image
        _, img_encoded = cv2.imencode(".png", processed)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        return https_fn.Response(
            json.dumps({"image": img_base64}),
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            status=200
        )

    except ImportError as e:
        return https_fn.Response(
            f"Dependency error: {str(e)}. Make sure all dependencies are in requirements.txt",
            status=500
        )
    except Exception as e:
        return https_fn.Response(
            f"Processing error: {str(e)}",
            status=500
        )
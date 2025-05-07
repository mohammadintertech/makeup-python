from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import io
from lipstick_processor import apply_lipstick
from eyeliner_processor import apply_eyeliner
from blusher_processor import apply_blusher
from foundation_processor import apply_foundation
from eyecolor_processor import apply_eyecolor
from eye_shadow_processor import apply_eyeshadow
import os

app = Flask(__name__)

@app.route("/apply", methods=["POST"])
def apply_makeup():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        contents = file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        processed = image.copy()

        form = request.form

        # Foundation
        if form.get('enable_foundation') == 'true':
            r, g, b = form.get('foundation_r'), form.get('foundation_g'), form.get('foundation_b')
            intensity = form.get('foundation_intensity')
            if all([r, g, b, intensity]):
                processed = apply_foundation(processed, [int(r), int(g), int(b)], float(intensity))

        # Blusher
        if form.get('enable_blusher') == 'true':
            r, g, b = form.get('blusher_r'), form.get('blusher_g'), form.get('blusher_b')
            intensity = form.get('blusher_intensity')
            if all([r, g, b, intensity]):
                processed = apply_blusher(processed, [int(r), int(g), int(b)], float(intensity))

        # Lipstick
        if form.get('enable_lipstick') == 'true':
            r, g, b = form.get('lipstick_r'), form.get('lipstick_g'), form.get('lipstick_b')
            intensity = form.get('lipstick_intensity')
            edge = form.get('lipstick_edge')
            if all([r, g, b, intensity, edge]):
                processed = apply_lipstick(processed, [int(r), int(g), int(b)], float(intensity), int(edge))

        # Eyeliner
        if form.get('enable_eyeliner') == 'true':
            r, g, b = form.get('eyeliner_r'), form.get('eyeliner_g'), form.get('eyeliner_b')
            if all([r, g, b]):
                processed = apply_eyeliner(processed, [int(r), int(g), int(b)])

        # Eye color
        if form.get('enable_eyecolor') == 'true':
            r, g, b = form.get('eyecolor_r'), form.get('eyecolor_g'), form.get('eyecolor_b')
            if all([r, g, b]):
                processed = apply_eyecolor(processed, int(r), int(g), int(b))

        # Eyeshadow
        if form.get('enable_eyeshadow') == 'true':
            r, g, b = form.get('eyeshadow_r'), form.get('eyeshadow_g'), form.get('eyeshadow_b')
            intensity = form.get('eyeshadow_intensity')
            if all([r, g, b, intensity]):
                processed = apply_eyeshadow(processed, [int(r), int(g), int(b)], float(intensity))

        _, img_encoded = cv2.imencode(".png", processed)
        return send_file(
            io.BytesIO(img_encoded.tobytes()),
            mimetype="image/png",
            as_attachment=False,
            download_name="result.png"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=8081)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

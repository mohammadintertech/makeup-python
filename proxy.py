# colab.py - PROXY RETURNING BINARY IMAGE DATA
from flask import Flask, request, jsonify, Response
import requests
import io
import logging
import os
from datetime import datetime
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Your Colab ngrok URL
COLAB_API_URL = "https://13ffae550e8f.ngrok-free.app"

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        response = requests.get(f"{COLAB_API_URL}/health", timeout=10)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": f"Cannot connect to Colab API: {str(e)}"}), 503

@app.route('/apply', methods=['POST'])
def apply_makeup():
    """
    Proxy endpoint - returns image as binary data for Flutter Uint8List
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Prepare data for forwarding to Colab
        files = {'file': (file.filename, file.stream, file.mimetype)}
        form_data = {}
        for key in request.form:
            form_data[key] = request.form[key]
        
        logger.info(f"Forwarding to Colab with {len(form_data)} parameters")
        
        # Forward to Colab
        response = requests.post(
            f"{COLAB_API_URL}/apply",
            files=files,
            data=form_data,
            timeout=60
        )
        
        # Process Colab response
        if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
            # Return binary image data directly
            return Response(
                response.content,
                mimetype='image/png',
                headers={
                    'Content-Type': 'image/png',
                    'Content-Disposition': 'inline; filename="makeup_result.png"'
                }
            )
        
        # If Colab returned an error
        error_response = response.json() if response.content else {"error": "Unknown error from Colab"}
        return jsonify(error_response), response.status_code
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "Colab API timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Colab API"}), 503
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        return jsonify({"error": f"Proxy error: {str(e)}"}), 500

@app.route('/apply_base64', methods=['POST'])
def apply_makeup_base64():
    """
    Alternative endpoint that returns base64 encoded image + JSON
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Prepare data for forwarding to Colab
        files = {'file': (file.filename, file.stream, file.mimetype)}
        form_data = {}
        for key in request.form:
            form_data[key] = request.form[key]
        
        # Forward to Colab
        response = requests.post(
            f"{COLAB_API_URL}/apply",
            files=files,
            data=form_data,
            timeout=60
        )
        
        # Process Colab response
        if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
            # Convert to base64
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            
            return jsonify({
                "success": True,
                "message": "Makeup applied successfully",
                "image_data": image_base64,
                "format": "base64",
                "size": len(response.content)
            })
        
        # If Colab returned an error
        error_response = response.json() if response.content else {"error": "Unknown error from Colab"}
        return jsonify(error_response), response.status_code
        
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        return jsonify({"error": f"Proxy error: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Proxy server is running", 
        "colab_url": COLAB_API_URL,
        "endpoints": {
            "POST /apply": "Returns binary image data (Uint8List)",
            "POST /apply_base64": "Returns base64 encoded image in JSON",
            "GET /health": "Health check"
        }
    })

@app.route('/')
def index():
    return jsonify({
        "message": "Makeup API Proxy Server",
        "status": "running",
        "colab_url": COLAB_API_URL
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
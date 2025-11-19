# colab.py - PROXY FOR MULTIPLE APIS
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import io
import logging
import os
from datetime import datetime
import base64
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API URLs
COLAB_API_URL = "https://makeup-python-5-6oru.onrender.com"
OASIS_API_URL = "https://www.oasisonline.ps/WS"

# ============================================================================
# AUTO-PING FUNCTION
# ============================================================================
def ping_colab_api():
    """Ping the Colab API every 10 minutes to keep it alive"""
    while True:
        try:
            time.sleep(600)  # 10 minutes = 600 seconds
            response = requests.get(f"{COLAB_API_URL}/ping", timeout=10)
            logger.info(f"Auto-ping to Colab API - Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Auto-ping failed: {str(e)}")

# Start the auto-ping thread
ping_thread = threading.Thread(target=ping_colab_api, daemon=True)
ping_thread.start()
logger.info("Auto-ping thread started - will ping Colab API every 10 minutes")

# ============================================================================
# COLAB MAKEUP API PROXY
# ============================================================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Colab"""
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

# ============================================================================
# OASIS ONLINE API PROXY
# ============================================================================
@app.route('/oasis/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def oasis_proxy(endpoint):
    """
    Generic proxy for Oasis Online API
    Example: /oasis/getProducts?c_id=123
    """
    try:
        # Build the full URL
        target_url = f"{OASIS_API_URL}/{endpoint}"
        
        # Prepare headers (you can modify as needed)
        headers = {
            'User-Agent': 'MakeupApp/1.0',
            'Accept': 'application/json',
        }
        
        # Forward the request
        if request.method == 'GET':
            response = requests.get(
                target_url,
                params=request.args,
                headers=headers,
                timeout=30
            )
        elif request.method == 'POST':
            response = requests.post(
                target_url,
                json=request.get_json(),
                data=request.form,
                headers=headers,
                timeout=30
            )
        elif request.method == 'PUT':
            response = requests.put(
                target_url,
                json=request.get_json(),
                headers=headers,
                timeout=30
            )
        elif request.method == 'DELETE':
            response = requests.delete(
                target_url,
                headers=headers,
                timeout=30
            )
        else:
            return jsonify({"error": "Method not allowed"}), 405
        
        # Return the response from Oasis API
        try:
            # Try to parse as JSON
            return jsonify(response.json()), response.status_code
        except:
            # Return as text if not JSON
            return Response(response.text, status=response.status_code, mimetype='text/plain')
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Oasis API timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Oasis API"}), 503
    except Exception as e:
        logger.error(f"Oasis proxy error: {str(e)}")
        return jsonify({"error": f"Oasis proxy error: {str(e)}"}), 500

# Specific Oasis endpoints for convenience
@app.route('/getProducts', methods=['GET'])
def get_products():
    """
    Specific endpoint for getProducts
    Usage: /getProducts?c_id=123
    """
    return oasis_proxy('getProducts')

@app.route('/getCategories', methods=['GET'])
def get_categories():
    """Proxy for getCategories endpoint"""
    return oasis_proxy('getCategories')

@app.route('/getProductDetails', methods=['GET'])
def get_product_details():
    """Proxy for getProductDetails endpoint"""
    return oasis_proxy('getProductDetails')

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Multi-API Proxy Server is running", 
        "colab_url": COLAB_API_URL,
        "oasis_url": OASIS_API_URL,
        "endpoints": {
            "Makeup API": {
                "POST /apply": "Returns binary image data (Uint8List)",
                "POST /apply_base64": "Returns base64 encoded image in JSON",
                "GET /health": "Health check"
            },
            "Oasis API": {
                "GET /getProducts": "Get products by category",
                "GET /getCategories": "Get categories",
                "GET /getProductDetails": "Get product details",
                "ANY /oasis/<endpoint>": "Generic Oasis API proxy"
            }
        }
    })

@app.route('/')
def index():
    return jsonify({
        "message": "Multi-API Proxy Server",
        "status": "running",
        "apis": {
            "makeup": COLAB_API_URL,
            "oasis": OASIS_API_URL
        },
        "usage": "Use /test endpoint for full endpoint list"
    })

# ... your existing code ...
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

# Add this line for Passenger
application = app
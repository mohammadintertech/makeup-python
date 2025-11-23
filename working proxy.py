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

# Track if we've already pinged
_ping_executed = False

# ============================================================================
# PING FUNCTION
# ============================================================================
def ping_colab_api_async():
    """Ping the Colab API in background without waiting for response"""
    def ping_task():
        try:
            response = requests.get(f"{COLAB_API_URL}/ping", timeout=5)
            logger.info(f"Ping to Colab API - Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Ping failed: {str(e)}")
    
    # Start ping in background thread
    ping_thread = threading.Thread(target=ping_task, daemon=True)
    ping_thread.start()

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

# ============================================================================
# MAKEUP PRODUCTS API PROXY
# ============================================================================
@app.route('/getProductsAPP', methods=['GET'])
def get_products_app():
    """
    Proxy for the new makeup products API
    Usage: /getProductsAPP?limit=10&page=1&c_id=136,72,78,84,87,137,139,138&makeup=1
    """
    try:
        # Ping Colab API in background (don't wait for response)
        ping_colab_api_async()
        
        # Build the target URL for the new API
        target_url = f"{OASIS_API_URL}/getProductsAPP"
        
        # Prepare query parameters
        params = {
            'limit': request.args.get('limit', '100'),
            'page': request.args.get('page', '1'),
            'c_id': request.args.get('c_id', '136,72,78,84,87,137,139,138'),
            'makeup': request.args.get('makeup', '1')
        }
        
        # Add any additional parameters
        for key in request.args:
            if key not in ['limit', 'page', 'c_id', 'makeup']:
                params[key] = request.args.get(key)
        
        logger.info(f"Fetching makeup products with params: {params}")
        
        # Forward the request to Oasis API
        response = requests.get(
            target_url,
            params=params,
            headers={
                'User-Agent': 'MakeupApp/1.0',
                'Accept': 'application/json',
            },
            timeout=30
        )
        
        # Process response
        if response.status_code == 200:
            try:
                data = response.json()
                logger.info(f"Successfully fetched {len(data.get('catproducts', []))} categories")
                return jsonify(data)
            except ValueError as e:
                logger.error(f"JSON parse error: {e}")
                return jsonify({"error": "Invalid JSON response from API"}), 500
        else:
            logger.error(f"API returned status {response.status_code}: {response.text}")
            return Response(response.text, status=response.status_code, mimetype='text/plain')
            
    except requests.exceptions.Timeout:
        logger.error("Timeout fetching makeup products")
        return jsonify({"error": "Oasis API timeout"}), 504
    except requests.exceptions.ConnectionError:
        logger.error("Connection error fetching makeup products")
        return jsonify({"error": "Cannot connect to Oasis API"}), 503
    except Exception as e:
        logger.error(f"Makeup products proxy error: {str(e)}")
        return jsonify({"error": f"Makeup products proxy error: {str(e)}"}), 500

@app.route('/getMakeupProducts', methods=['GET'])
def get_makeup_products():
    """
    Convenience endpoint specifically for makeup products with optimized defaults
    Usage: /getMakeupProducts?limit=50
    """
    try:
        # Use optimized defaults for makeup products
        params = {
            'limit': request.args.get('limit', '50'),
            'page': request.args.get('page', '1'),
            'c_id': '136,72,78,84,87,137,139,138',  # Makeup categories
            'makeup': '1'
        }
        
        # Create a new request with these parameters
        request.args = type('Args', (), {**params, 'get': lambda self, key, default=None: params.get(key, default)})()
        
        # Call the main proxy function
        return get_products_app()
        
    except Exception as e:
        logger.error(f"Makeup products convenience endpoint error: {str(e)}")
        return jsonify({"error": f"Error fetching makeup products: {str(e)}"}), 500

# ============================================================================
# LEGACY OASIS ENDPOINTS (for backward compatibility)
# ============================================================================
@app.route('/getProducts', methods=['GET'])
def get_products():
    """
    Specific endpoint for getProducts (legacy)
    Usage: /getProducts?c_id=123
    """
    # Ping Colab API in background (don't wait for response)
    ping_colab_api_async()
    
    # Continue with the original Oasis API call
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
        "status": "healthy"
    })

@app.route('/')
def index():
    return jsonify({
        "message": "API Proxy Server",
        "status": "running"
    })

# ============================================================================
# APPLICATION STARTUP - FIXED FOR NEW FLASK VERSION
# ============================================================================
@app.before_request
def before_first_request():
    """Ping Colab API on first request - fixed for new Flask version"""
    global _ping_executed
    if not _ping_executed:
        logger.info("Starting up Multi-API Proxy Server - First request")
        ping_colab_api_async()
        _ping_executed = True

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

# Add this line for Passenger
application = app
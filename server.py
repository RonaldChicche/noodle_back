import os
import cv2
import base64
import numpy as np
import procesador as pr
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app) 
procesador = pr.Procesador()
model_path = procesador.weights_path['epoch_100']['best']
procesador.load_model(model_path)
HOME_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HOME_PATH, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/')
def index():
    return 'Hello, World!'

# Test the server
@app.route('/test_connection', methods=['GET'])
def test():
    return jsonify({'test': 'success'})


@app.route('/test_imagen', methods=['GET'])
def servir_imagen():
    # Genera una imagen con openCV
    image = np.zeros((100, 100, 3), np.uint8)
    image[:] = (255, 0, 0)
    # Codificar la imagen en base64
    retval, buffer = cv2.imencode('.jpg', image)
    image_encoded = base64.b64encode(buffer)
    # Crear la respuesta
    response = {
        'success': True,
        'image': image_encoded.decode('utf-8')
    }
    # Enviar la imagen como respuesta
    return jsonify(response)

@app.route('/check_image', methods=['POST'])
def check_image():
    data = request.files['image']
    # red the image
    image = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
    size = image.shape
    # return the size of the image
    response = {
        'check': '1' if size else '0',
        'size': size
    }
    
    return jsonify(response)

@app.route('/yolo_predict', methods=['POST'])
def yolo_predict():
    # try:
    # Verifica si se ha enviado un archivo
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    
    data = request.files['image']
    image = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)

    img, results = procesador.score_frame(image)
    detections_dict = {
        "boxes": results.xyxy.tolist(),  # Convertir a lista si es un ndarray
        "class_ids": results.class_id.tolist(),  # Convertir a lista si es un ndarray
        "confidences": results.confidence.tolist()  # Convertir a lista si es un ndarray
    }


    # save the image with output + the current data as part of its name
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    image_name = f"image_{current_time}.jpg"
    image_path = os.path.join(HOME_PATH, 'output', image_name)
    print(image_path)
    cv2.imwrite(image_path, img)
    # convert locatio to url
    image_url = f"http://localhost:5000/output/{image_name}"
    print(image_url)
    
        
    # Codifica la imagen procesada en base64
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer)

    response = {
        'success': True,
        'results': detections_dict,
        'image': image_url
    }
    # except:
    #     response = {
    #         'success': False,
    #         'results': [],
    #         'image': ''
    #     }

    return jsonify(response)

@app.route('/models', methods=['GET'])
def models():
    return jsonify(procesador.weights_path)

@app.route('/output/<path:filename>')
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



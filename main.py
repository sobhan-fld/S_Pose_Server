from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from datetime import datetime
import json
import time
import S_Pose
# from flask_cors import CORS
from flask import send_from_directory


app = Flask(__name__)
# CORS(app)

@app.route('/butterfly', methods=['POST'])
def route_butterfly():
    # Decode the image from base64 format
    image_data = request.form['image'].split(',', 1)[1]
    decoded_image = base64.b64decode(image_data)
    image_np = np.frombuffer(decoded_image, dtype=np.uint8)
    frame = cv2.imdecode(image_np, flags=1)
    # Process the frame
    frame, angles_dict = S_Pose.butterfly(frame)

    # Encode the processed frame back to send to the client
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': "data:image/jpeg;base64," + frame_encoded,
        'angles': angles_dict
    })

@app.route('/squat', methods=['POST'])
def route_squat():
    # Decode the image from base64 format
    image_data = request.form['image'].split(',', 1)[1]
    decoded_image = base64.b64decode(image_data)
    image_np = np.frombuffer(decoded_image, dtype=np.uint8)
    frame = cv2.imdecode(image_np, flags=1)
    # Process the frame
    frame, angles_dict = S_Pose.squat(frame)

    # Encode the processed frame back to send to the client
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': "data:image/jpeg;base64," + frame_encoded,
        'angles': angles_dict
    })

@app.route('/legtrain', methods=['POST'])
def route_legtrain():
    # Decode the image from base64 format
    image_data = request.form['image'].split(',', 1)[1]
    decoded_image = base64.b64decode(image_data)
    image_np = np.frombuffer(decoded_image, dtype=np.uint8)
    frame = cv2.imdecode(image_np, flags=1)
    # Process the frame
    frame, angles_dict = S_Pose.legtrain(frame)

    # Encode the processed frame back to send to the client
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': "data:image/jpeg;base64," + frame_encoded,
        'angles': angles_dict
    })

@app.route('/curling', methods=['POST'])
def route_curling():
    # Decode the image from base64 format
    image_data = request.form['image'].split(',', 1)[1]
    decoded_image = base64.b64decode(image_data)
    image_np = np.frombuffer(decoded_image, dtype=np.uint8)
    frame = cv2.imdecode(image_np, flags=1)
    # Process the frame
    frame, angles_dict = S_Pose.curling(frame)

    # Encode the processed frame back to send to the client
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': "data:image/jpeg;base64," + frame_encoded,
        'angles': angles_dict
    })

@app.route('/hand', methods=['POST'])
def route_hand():
    # Decode the image from base64 format
    image_data = request.form['image'].split(',', 1)[1]
    decoded_image = base64.b64decode(image_data)
    image_np = np.frombuffer(decoded_image, dtype=np.uint8)
    frame = cv2.imdecode(image_np, flags=1)
    # Process the frame
    frame = S_Pose.hand(frame)

    # Encode the processed frame back to send to the client
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': "data:image/jpeg;base64," + frame_encoded,
    })

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0' ,debug=True, port=5000)

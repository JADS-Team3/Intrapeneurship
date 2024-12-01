from flask import Flask, render_template, request, jsonify
import os
import time  # Import time module for delay
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import cv2
from kneed import KneeLocator  # Ensure you have installed kneed via pip
from sklearn_extra.cluster import KMedoids
import random
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import math
import matplotlib.image as mpimg
import prism_script as prism
import matplotlib.image as mpimg
import math

# Flask app configuration
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to get the next house name
def get_next_house_name():
    existing_houses = os.listdir(UPLOAD_FOLDER)
    house_numbers = [int(name.split()[-1]) for name in existing_houses if name.startswith("House")]
    next_number = max(house_numbers, default=0) + 1
    return f"House {next_number}"

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# Upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        
        method = request.form.get("method") if request.form.get("method") is not None else "Not specified"
        description = request.form.get("description") if request.form.get("description") is not None else "No description provided"
        title = request.form.get("title") if request.form.get("title") is not None else f"House {len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1}"
        if title == "":
            title = f"House {len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1}"

        # Create a folder for this upload
        house_id = f"{title}"
        house_folder = os.path.join(app.config['UPLOAD_FOLDER'], house_id)
        os.makedirs(house_folder, exist_ok=True)

        # Save all uploaded images to the folder
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(house_folder, filename))

        clusters = 0
        return jsonify({
            "files": len(files),
            "method": method,
            "description": description,
            "title": title,
            "clusters": clusters
            }), 200
    return render_template('upload.html', loading=False)

# Prism page
@app.route('/prism', methods=['POST'])
def prism():
    data = request.get_json()
    clusters = data.get('clusters')
    title = data.get('title')
    description = data.get('description')
    method = data.get('method')

    # Dummy response for testing
    response = {
        "message": "Prism request received",
        "clusters": clusters,
        "title": title,
        "description": description,
        "method": method
    }
    return jsonify(response), 200

# Gallery page
@app.route('/gallery')
def gallery():
    houses = os.listdir(UPLOAD_FOLDER)
    return render_template('gallery.html', houses=houses)

# House details page
@app.route('/gallery/<house_name>')
def house(house_name):
    house_folder = os.path.join(app.config['UPLOAD_FOLDER'], house_name)
    if not os.path.exists(house_folder):
        return "House not found", 404

    images = [f"uploads/{house_name}/{img}" for img in os.listdir(house_folder) if allowed_file(img)]
    num_images = len(images)
    total_size_bytes = sum(
        os.path.getsize(os.path.join(house_folder, img)) for img in os.listdir(house_folder) if allowed_file(img)
    )
    total_size_mb = total_size_bytes / (1024 * 1024)  # Convert bytes to MB

    return render_template(
        'house.html',
        house_name=house_name,
        images=images,
        num_images=num_images,
        total_size_mb=total_size_mb
    )

if __name__ == '__main__':
    app.run(debug=True)
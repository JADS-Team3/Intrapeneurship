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
import json
from kneed import KneeLocator  # Ensure you have installed kneed via pip
from sklearn_extra.cluster import KMedoids
import random
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import math
import matplotlib.image as mpimg
from prism_script import (
    extract_features as prism_extract_features,
    normalize_features as prism_normalize_features,
    find_optimal_clusters_combined as prism_find_optimal_clusters_combined,
    kmedoids_clustering as prism_kmedoids_clustering,
    kmeans_clustering as prism_kmeans_clustering,
    select_representative_images_laplacian as prism_select_representative_images_laplacian,
    select_representative_images_centroid as prism_select_representative_images_centroid,
    select_representative_images_medoid as prism_select_representative_images_medoid
)

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

def calculate_recommended_clusters(image_folder):
    """Calculate the recommended number of clusters for a given house
        # Step 1: Extract features and compute quality scores
        # Step 2: Normalize features
        # Step 3: Find the optimal number of clusters using combined method
        Returns:
            clusters: Recommended number of clusters"""

    # Step 1: Extract features and compute quality scores
    features, img_paths, quality_scores = prism_extract_features(image_folder)
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of images processed: {len(img_paths)}")

    if len(features) == 0:
        IndexError("No images were processed. Please check the image folder path and contents.")
    # Step 2: Normalize features
    features_normalized = prism_normalize_features(features)
    

    # Step 3: Find the optimal number of clusters using combined method
    # Step 3: Find the optimal number of clusters using combined method
    optimal_k, silhouette_scores, wcss = prism_find_optimal_clusters_combined(
        features_normalized, min_clusters=2, max_clusters=len(img_paths) - 1, silhouette_threshold=0.001
    )
    return optimal_k

def calculate_clusters(image_folder, selection_method, clusters):
    """Calculate the recommended number of clusters for a given house
        # Step 3: Cluster images using the appropriate clustering algorithm
        # Step 4: Select and print representative images based on the chosen method
        Returns:
            dict: image names for each cluster"""

    # Step 1: Extract features and compute quality scores
    features, img_paths, quality_scores = prism_extract_features(image_folder)

    if len(features) == 0:
        IndexError("No images were processed. Please check the image folder path and contents.")

    # Step 2: Normalize features
    features_normalized = prism_normalize_features(features)

    # Step 3: Cluster images using the appropriate clustering algorithm
    if selection_method == 'medoid':
        kmedoids = prism_kmedoids_clustering(features_normalized, n_clusters=clusters)
        labels = kmedoids.labels_
        centroids = kmedoids.cluster_centers_
    else:
        kmeans = prism_kmeans_clustering(features_normalized, n_clusters=clusters)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
    # Step 4: Select representative images based on the chosen method
    if selection_method == 'laplacian':
        representative_img, cluster_images_dict = prism_select_representative_images_laplacian(
            labels, img_paths, quality_scores
        )
    elif selection_method == 'centroid':
        representative_img, cluster_images_dict = prism_select_representative_images_centroid(
            labels, img_paths, features_normalized, centroids
        )
    elif selection_method == 'medoid':
        representative_img, cluster_images_dict = prism_select_representative_images_medoid(
            kmedoids, img_paths
        )
    else:
        IndexError("Invalid selection method selected.")

    # Step 5: Change the order of the cluster_images_dict, for each cluster the first image is the representative image
    representative_img = [v for k, v in representative_img.items()]
    print("Representative images:", representative_img)
    print("Cluster images dict:", cluster_images_dict)
    # Change the order of the keys inside the cluster_images_dict
    new_cluster_images_dict = {}

    for i, (cluster, images) in enumerate(cluster_images_dict.items()):
        if i < len(representative_img):
            rep_img = representative_img[i]
            if rep_img in images:
                images.remove(rep_img)
            images.insert(0, rep_img)
        new_cluster_images_dict[cluster] = images

    print("Cluster images dict:", cluster_images_dict)    
    
    
    return cluster_images_dict
# Main page
@app.route('/')
def index():
    return render_template('index.html')

# Upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload a new house with images and metadata
        Returns:
            metadata (len(files), method, description, title, clusters)
            clusters: Number of recommended clusters"""
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

        clusters = calculate_recommended_clusters(house_folder)
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
    """"Receive a request to run PRISM on a set of images
        Returns:
            clusters: Number of recommended clusters
            title: Title of the house
            description: Description of the house
            method: Method used for clustering"""
    data = request.get_json()
    clusters = int(data.get('clusters'))
    title = data.get('title')
    house_id = f"{title}"
    house_folder = os.path.join(app.config['UPLOAD_FOLDER'], house_id)
    description = data.get('description')
    method = data.get('method')
    cluster_dict = calculate_clusters(house_folder, method, clusters)

    response = {
        "clusters": clusters,
        "title": title,
        "description": description,
        "method": method,
        "cluster_dict": {str(k): v for k, v in cluster_dict.items()}
    }

    # Create new json file inside the house folder with the response
    with open(os.path.join(house_folder, 'prism.json'), 'w') as f:
        f.write(json.dumps(response, indent=4))
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

    # Check if the house exists
    if not os.path.exists(house_folder):
        return "House not found", 404
    
    # Make sure PRISM has been run on this house
    if not os.path.exists(os.path.join(house_folder, 'prism.json')):
        return "PRISM not run on this house yet", 404

    # Load the PRISM results
    with open(os.path.join(house_folder, 'prism.json')) as f:
        prism_results = json.load(f)
        house_name = prism_results['title']
        description = prism_results['description']
        method = prism_results['method']
        clusters = prism_results['clusters']
        cluster_dict = prism_results['cluster_dict']

    images = [f"uploads/{house_name}/{img}" for img in os.listdir(house_folder) if allowed_file(img)]
    num_images = len(images)
    total_size_bytes = sum(
        os.path.getsize(os.path.join(house_folder, img)) for img in os.listdir(house_folder) if allowed_file(img)
    )
    total_size_mb = total_size_bytes / (1024 * 1024)  # Convert bytes to MB

    return render_template(
        'house.html',
        house_name=house_name,
        description=description,
        clusters=clusters,
        cluster_dict=cluster_dict,
        method=method,
        num_images=num_images,
        total_size_mb=total_size_mb
    )

if __name__ == '__main__':
    app.run(debug=True)
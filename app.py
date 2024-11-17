from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.cluster import KMeans
import PIL

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

# Function to load and preprocess an image
def load_and_preprocess(img_path):
    try:
        print(f"Loading image: {img_path}")  # Debugging log
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to ResNet50 input size
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = preprocess_input(x)  # Apply ResNet50 preprocessing
        return x
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")  # Debugging log
        return None

# Extract features from images
def extract_features(image_folder):
    features = []
    img_paths = []
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        print(f"Processing image: {img_path}")  # Debugging log
        img = load_and_preprocess(img_path)
        if img is not None:
            try:
                feature = model.predict(img)
                features.append(feature.flatten())
                img_paths.append(img_path)
            except Exception as e:
                print(f"Error extracting features from {img_path}: {e}")  # Debugging log
    print(f"Features extracted: {len(features)} images processed.")  # Summary log
    return np.array(features), img_paths


# Normalize features
def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Perform KMeans clustering
def kmeans_clustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans.labels_

# Process uploaded images with KMeans
def process_images_with_kmeans(house_folder, n_clusters):
    # Extract features from images
    features, img_paths = extract_features(house_folder)
    
    if len(features) == 0:
        print('no features')
        return  # No images to process

    # Normalize features
    features_normalized = normalize_features(features)

    # Perform clustering
    labels = kmeans_clustering(features_normalized, n_clusters=n_clusters)
    print(labels)
    # Keep one representative image per cluster
    cluster_representatives = {}
    for label, img_path in zip(labels, img_paths):
        if label not in cluster_representatives:
            cluster_representatives[label] = img_path  # First image in the cluster

    # Remove unnecessary images
    for img_path in img_paths:
        if img_path not in cluster_representatives.values():
            print("removes"+img_path)
            os.remove(img_path)

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
            return "No files uploaded", 400

        # Get the number of clusters from the form
        n_clusters = int(request.form.get('clusters', 5))  # Default to 5 if not provided

        # Create a folder for this upload
        house_id = f"House {len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1}"
        house_folder = os.path.join(app.config['UPLOAD_FOLDER'], house_id)
        os.makedirs(house_folder, exist_ok=True)

        # Save all uploaded images to the folder
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(house_folder, filename))

        # Process images using KMeans to remove unnecessary ones
        process_images_with_kmeans(house_folder, n_clusters=n_clusters)

        return redirect(url_for('gallery'))

    return render_template('upload.html')


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

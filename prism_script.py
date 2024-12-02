import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
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

def load_and_preprocess(img_path):
    """
    Loads and preprocesses an image for feature extraction.
    
    Parameters:
    - img_path (str): Path to the image file.
    
    Returns:
    - np.ndarray or None: Preprocessed image array or None if loading fails.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

# --------------------------- Quality Score Computations ---------------------------

def compute_laplacian_variance(img_path):
    """
    Computes the variance of the Laplacian of the image to assess sharpness.
    
    Parameters:
    - img_path (str): Path to the image file.
    
    Returns:
    - float: Variance of Laplacian score.
    """
    try:
        image_cv = cv2.imread(img_path)
        if image_cv is None:
            print(f"Unable to read image {img_path}. Returning quality score of 0.")
            return 0.0
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    except Exception as e:
        print(f"Error computing Laplacian for {img_path}: {e}")
        return 0.0

# --------------------------- Feature Extraction ---------------------------

def extract_features(image_folder):
    """
    Extracts features from images and computes their quality scores.
    
    Parameters:
    - image_folder (str): Path to the folder containing images.
    
    Returns:
    - np.ndarray: Array of extracted features.
    - list: List of image paths.
    - list: List of quality scores corresponding to each image.
    """
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    features = []
    img_paths = []
    quality_scores = []
    for img_name in tqdm(os.listdir(image_folder), desc="Extracting Features"):
        img_path = os.path.join(image_folder, img_name)
        if not os.path.isfile(img_path):
            continue  # Skip if not a file
        img = load_and_preprocess(img_path)
        if img is not None:
            feature = model.predict(img)
            features.append(feature.flatten())
            img_paths.append(img_path)
            quality = compute_laplacian_variance(img_path)  # Default quality metric
            quality_scores.append(quality)
    return np.array(features), img_paths, quality_scores

# --------------------------- Feature Normalization ---------------------------

def normalize_features(features):
    """
    Normalizes the feature vectors using StandardScaler.
    
    Parameters:
    - features (np.ndarray): Array of image features.
    
    Returns:
    - np.ndarray: Normalized feature array.
    """
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    return normalized

# --------------------------- Representative Selection Methods ---------------------------

def select_representative_images_laplacian(labels, img_paths, quality_scores):
    """
    Selects representative images based on the highest Laplacian quality score within each cluster.
    
    Parameters:
    - labels (np.ndarray): Cluster labels for each image.
    - img_paths (list): List of image paths.
    - quality_scores (list): List of Laplacian quality scores for each image.
    
    Returns:
    - dict: Mapping from cluster label to representative image path.
    - dict: Mapping from cluster label to list of image paths in the cluster.
    """
    clusters = {}
    cluster_images = {}

    # Organize images by cluster
    for label, img_path, quality in zip(labels, img_paths, quality_scores):
        if label == -1:
            continue  # Noise (if applicable)
        if label not in cluster_images:
            cluster_images[label] = []
        cluster_images[label].append((img_path, quality))
    
    # For each cluster, find the image with the highest quality score
    for label, images in cluster_images.items():
        # Sort images in descending order based on quality score
        sorted_images = sorted(images, key=lambda x: x[1], reverse=True)
        # Select the image with the highest quality score as the representative
        clusters[label] = sorted_images[0][0]
    
    # Also gather all images in each cluster
    cluster_images_dict = {label: [img for img, _ in images] for label, images in cluster_images.items()}
    
    return clusters, cluster_images_dict

def select_representative_images_centroid(labels, img_paths, features, centroids):
    """
    Selects representative images based on proximity to the cluster centroid.
    
    Parameters:
    - labels (np.ndarray): Cluster labels for each image.
    - img_paths (list): List of image paths.
    - features (np.ndarray): Array of image features.
    - centroids (np.ndarray): Centroids of the clusters.
    
    Returns:
    - dict: Mapping from cluster label to representative image path.
    - dict: Mapping from cluster label to list of image paths in the cluster.
    """
    clusters = {}
    cluster_images = {}

    # Organize images by cluster
    for label, img_path in zip(labels, img_paths):
        if label == -1:
            continue  # Noise (if applicable)
        if label not in cluster_images:
            cluster_images[label] = []
        cluster_images[label].append(img_path)
    
    # For each cluster, find the image closest to the centroid
    for label, images in cluster_images.items():
        # Extract features of images in the current cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_features = features[cluster_indices]
        centroid = centroids[label].reshape(1, -1)
        distances = cdist(cluster_features, centroid, 'euclidean').flatten()
        # Find the index of the image closest to the centroid
        closest_idx = np.argmin(distances)
        representative_img = images[closest_idx]
        clusters[label] = representative_img
    
    return clusters, cluster_images

def select_representative_images_medoid(kmedoids, img_paths):
    """
    Selects representative images based on K-Medoids clustering.
    
    Parameters:
    - kmedoids (KMedoids): Trained KMedoids object.
    - img_paths (list): List of image paths.
    
    Returns:
    - dict: Mapping from cluster label to representative image path.
    - dict: Mapping from cluster label to list of image paths in the cluster.
    """
    clusters = {}
    cluster_images = {}

    for label, img_path in zip(kmedoids.labels_, img_paths):
        if label not in cluster_images:
            cluster_images[label] = []
        cluster_images[label].append(img_path)
    
    for label, medoid_idx in enumerate(kmedoids.medoid_indices_):
        clusters[label] = img_paths[medoid_idx]
    
    return clusters, cluster_images

# --------------------------- Cluster Visualization ---------------------------

def reduce_dimensionality(features, method='pca', n_components=2, perplexity=30, n_iter=300):
    """
    Reduces feature dimensionality for visualization.
    
    Parameters:
    - features (np.ndarray): Array of image features.
    - method (str): 'pca' or 'tsne'.
    - n_components (int): Number of dimensions.
    - perplexity (int): Perplexity parameter for t-SNE.
    - n_iter (int): Number of iterations for t-SNE.
    
    Returns:
    - np.ndarray: Reduced feature array.
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(features)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
        reduced = reducer.fit_transform(features)
    else:
        raise ValueError("Unsupported dimensionality reduction method.")
    return reduced

def plot_clusters(reduced_features, labels, title='Clusters', dimensionality=2):
    """
    Plots the clusters in a 2D or 3D space.
    
    Parameters:
    - reduced_features (np.ndarray): 2D or 3D array of reduced features.
    - labels (np.ndarray): Cluster labels.
    - title (str): Title of the plot.
    - dimensionality (int): 2 for 2D plot, 3 for 3D plot.
    """
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    if dimensionality == 2:
        plt.figure(figsize=(10, 7))
        for label in unique_labels:
            if label == -1:
                # Noise
                color = 'k'
                marker = 'x'
                label_name = 'Noise'
            else:
                color = colors(label)
                marker = 'o'
                label_name = f'Cluster {label}'
            plt.scatter(
                reduced_features[labels == label, 0],
                reduced_features[labels == label, 1],
                c=[color],
                marker=marker,
                label=label_name,
                s=10
            )
        plt.title(title)
        plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    elif dimensionality == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        for label in unique_labels:
            if label == -1:
                # Noise
                color = 'k'
                marker = 'x'
                label_name = 'Noise'
            else:
                color = colors(label)
                marker = 'o'
                label_name = f'Cluster {label}'
            ax.scatter(
                reduced_features[labels == label, 0],
                reduced_features[labels == label, 1],
                reduced_features[labels == label, 2],
                c=[color],
                marker=marker,
                label=label_name,
                s=20,
                alpha=0.6
            )
        ax.set_title(title)
        ax.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.tight_layout()
        plt.show()
    
    else:
        print("Unsupported dimensionality for plotting. Choose 2 or 3.")

# --------------------------- Clustering Functions ---------------------------

def kmeans_clustering(features, n_clusters):
    """
    Performs K-Means clustering.
    
    Parameters:
    - features (np.ndarray): Array of image features.
    - n_clusters (int): Number of clusters.
    
    Returns:
    - KMeans: Trained KMeans object.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

def kmedoids_clustering(features, n_clusters):
    """
    Performs K-Medoids clustering.
    
    Parameters:
    - features (np.ndarray): Array of image features.
    - n_clusters (int): Number of clusters.
    
    Returns:
    - KMedoids: Trained KMedoids object.
    """
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, metric='euclidean')
    kmedoids.fit(features)
    return kmedoids

def find_optimal_clusters_combined(features, min_clusters=2, max_clusters=30, silhouette_threshold=0.2):
    """
    Determines the optimal number of clusters using Silhouette and Elbow methods.
    
    Parameters:
    - features (np.ndarray): Array of image features.
    - min_clusters (int): Minimum number of clusters to try.
    - max_clusters (int): Maximum number of clusters to try.
    - silhouette_threshold (float): Minimum acceptable silhouette score.
    
    Returns:
    - int: Optimal number of clusters.
    - list: Silhouette scores for each k.
    - list: WCSS values for each k.
    """
    silhouette_scores = []
    wcss = []
    K = range(min_clusters, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        print(labels.shape, features.shape)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
    
    # Find the elbow point
    kl = KneeLocator(K, wcss, curve='convex', direction='decreasing')
    elbow_k = kl.elbow if kl.elbow else max_clusters  # If elbow not found, use max_clusters
    
    # Choose the highest k where silhouette score is above the threshold and <= elbow_k
    acceptable_k = [k for k, score in zip(K, silhouette_scores) if score >= silhouette_threshold]
    if not acceptable_k:
        optimal_k = elbow_k
    else:
        # Choose the largest acceptable k up to the elbow point
        acceptable_k = [k for k in acceptable_k if k <= elbow_k]
        if acceptable_k:
            optimal_k = max(acceptable_k)
        else:
            optimal_k = elbow_k

    return optimal_k, silhouette_scores, wcss

# --------------------------- Cluster Display Function ---------------------------

import matplotlib.image as mpimg
import math

def display_selected_clusters(clusters, cluster_images_dict, selected_labels):
    """
    Displays the representative image and all images for selected clusters.

    Parameters:
    - clusters (dict): Mapping from cluster label to representative image path.
    - cluster_images_dict (dict): Mapping from cluster label to list of image paths.
    - selected_labels (list): List of cluster labels to display.
    """
    for label in selected_labels:
        print(f"\nCluster {label}:")
        
        # Check if the label exists
        if label not in clusters:
            print(f"Cluster {label} does not exist.")
            continue
        
        representative_img = clusters[label]
        cluster_imgs = cluster_images_dict.get(label, [])
        
        # Check if there are images in the cluster
        if not cluster_imgs:
            print(f"Cluster {label} has no images.")
            continue
        
        # Define the total number of images
        total_images = len(cluster_imgs)
        
        # Calculate grid size (e.g., maximum 5 columns)
        cols = min(5, total_images)
        rows = math.ceil(total_images / cols)
        
        # Create a figure with a grid of subplots
        fig_width = 3 * cols
        fig_height = 3 * rows
        fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        axs = axs.flatten()  # Flatten in case of multiple rows
        
        for idx, img_path in enumerate(cluster_imgs):
            ax = axs[idx]
            try:
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.axis('off')
                title = os.path.basename(img_path)
                if img_path == representative_img:
                    ax.set_title(f"Rep: {title}", fontsize=10, color='red')
                else:
                    ax.set_title(title, fontsize=8)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                ax.axis('off')
        
        # Hide any remaining subplots if the grid is larger than the number of images
        for j in range(idx + 1, len(axs)):
            axs[j].axis('off')
        
        plt.suptitle(f'Cluster {label} Visualization', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
        plt.show()

# --------------------------- Utility Functions ---------------------------

def print_clusters_and_representatives(clusters, cluster_images):
    """
    Prints the clusters and their representative images.
    
    Parameters:
    - clusters (dict): Mapping from cluster label to representative image path.
    - cluster_images (dict): Mapping from cluster label to list of image paths in the cluster.
    """
    for label in sorted(clusters.keys()):
        print(f"\nCluster {label}:")
        print(f"Representative image: {clusters[label]}")
        print("Images in cluster:")
        for img in cluster_images[label]:
            print(f" - {img}")
        print("-" * 50)

# --------------------------- Main Execution ---------------------------

def main():
    # Path to your images folder
    image_folder = 'images/'  # Replace with your actual image folder path
    
    # Selection Method Options
    selection_methods = {
        'laplacian': 'Image Quality Score (Laplacian)',
        'centroid': 'Closest to Cluster Centroid (Centroid Proximity)',
        'medoid': 'Medoid Selection (K-Medoids)'
    }
    
    # Define which selection method to use
    selection_method = 'laplacian'  # Options: 'laplacian', 'centroid', 'medoid'
    
    # Define plot dimensionality
    plot_dimension = 3  # Set to 2 for 2D plots, 3 for 3D plots
    
    if selection_method not in selection_methods:
        print(f"Invalid selection method. Choose from: {list(selection_methods.keys())}")
        return
    
    print(f"Selected Representative Selection Method: {selection_methods[selection_method]}")
    print(f"Selected Plot Dimensionality: {plot_dimension}D")
    
    # Step 1: Extract features and compute quality scores
    features, img_paths, quality_scores = extract_features(image_folder)
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of images processed: {len(img_paths)}")
    
    if len(features) == 0:
        print("No images were processed. Please check the image folder path and contents.")
        return
    
    # Step 2: Normalize features
    features_normalized = normalize_features(features)
    
    # Step 3: Find the optimal number of clusters using combined method
    optimal_k, silhouette_scores, wcss = find_optimal_clusters_combined(
        features_normalized, min_clusters=2, max_clusters=50, silhouette_threshold=0.001
    )
    print(f"Optimal number of clusters determined by combined method: {optimal_k}")
    
    # Plotting Silhouette Scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 51), silhouette_scores, marker='o')
    plt.title('Silhouette Scores vs Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plotting WCSS
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 51), wcss, marker='o')
    plt.title('Elbow Method - WCSS vs Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Step 4: Cluster images using the appropriate clustering algorithm
    if selection_method == 'medoid':
        kmedoids = kmedoids_clustering(features_normalized, n_clusters=optimal_k)
        labels = kmedoids.labels_
        centroids = kmedoids.cluster_centers_
    else:
        kmeans = kmeans_clustering(features_normalized, n_clusters=optimal_k)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    
    # Display clustering information
    num_clusters = optimal_k
    print(f"Number of clusters: {num_clusters}")
    
    # Step 5: Visualize clusters (2D or 3D)
    reduced_features = reduce_dimensionality(
        features_normalized,
        method='pca',  # You can switch to 'tsne' if preferred
        n_components=plot_dimension,
        perplexity=30,
        n_iter=300
    )
    plot_clusters(reduced_features, labels, title='Clusters Visualization with PCA', dimensionality=plot_dimension)
    
    # Step 6: Select representative images based on the chosen method
    if selection_method == 'laplacian':
        clusters, cluster_images_dict = select_representative_images_laplacian(
            labels, img_paths, quality_scores
        )
    elif selection_method == 'centroid':
        clusters, cluster_images_dict = select_representative_images_centroid(
            labels, img_paths, features_normalized, centroids
        )
    elif selection_method == 'medoid':
        clusters, cluster_images_dict = select_representative_images_medoid(
            kmedoids, img_paths
        )
    else:
        print("Invalid selection method selected.")
        return
    
    print("Representative images and their clusters:")
    print_clusters_and_representatives(clusters, cluster_images_dict)
    
    # Step 7: Display selected clusters
    # Define which clusters to display
    # For example, you can display the first 3 clusters
    selected_cluster_labels = list(clusters.keys())[:3]  # Modify as needed (e.g., [0], [1, 3, 5])
    
    # Ensure that the selected labels exist
    existing_labels = set(labels)
    valid_selected_labels = [label for label in selected_cluster_labels if label in existing_labels]
    
    if valid_selected_labels:
        display_selected_clusters(clusters, cluster_images_dict, valid_selected_labels)
    else:
        print("No valid cluster labels selected for plotting.")
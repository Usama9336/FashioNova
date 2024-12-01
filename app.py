import os
import pickle
import numpy as np
from numpy.linalg import norm
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
import cloudinary
import cloudinary.uploader
import cloudinary.api
from pymongo import MongoClient
from PIL import Image as pil_image
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Cloudinary configuration
cloudinary.config(
    cloud_name='dw6ujmsc5',  # Replace with your Cloudinary cloud name
    api_key='118658342512472',  # Replace with your Cloudinary API key
    api_secret='mRr4Zhd7zf51vrMkU79Ah1qlZJI'  # Replace with your Cloudinary API secret
)

# MongoDB connection
client = MongoClient('mongodb+srv://usama9336:usama@cluster0.0bdrk.mongodb.net/')
db = client['imagesearch']
image_collection = db['images']

# Load precomputed embeddings and filenames
with open('embeddings.pkl', 'rb') as f:
    precomputed_embeddings = pickle.load(f)
    
with open('filenames.pkl', 'rb') as f:
    precomputed_filenames = pickle.load(f)

# Convert to numpy arrays
precomputed_embeddings = np.array(precomputed_embeddings)
precomputed_filenames = np.array(precomputed_filenames)

# Load pre-trained ResNet50 model + GlobalMaxPooling2D
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Feature extraction function
def extract_features(img):
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Upload image, search for nearest images, and store in MongoDB
@app.route('/upload', methods=['POST'])
def upload_and_search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Upload the image to Cloudinary
        upload_result = cloudinary.uploader.upload(file)
        cloudinary_url = upload_result['secure_url']

        # Read and process the image for feature extraction
        file.seek(0)
        img = pil_image.open(file).resize((224, 224))
        feature_vector = extract_features(img)

        # Fetch all image embeddings and URLs from the database
        all_images = list(image_collection.find())
        embeddings = []
        cloudinary_urls = []

        for image_data in all_images:
            cloudinary_urls.append(image_data['cloudinary_url'])
            embeddings.append(pickle.loads(image_data['embedding']))

        embeddings = np.array(embeddings)

        # Nearest Neighbors search (if there are existing images in the database)
        nearest_images = []
        if len(embeddings) > 0:
            neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
            neighbors.fit(embeddings)

            # Find the nearest neighbors
            distances, indices = neighbors.kneighbors([feature_vector])

            # Retrieve the paths of the nearest images
            nearest_image_paths = [all_images[idx]['filename'] for idx in indices[0][1:6]]  # Exclude the uploaded image
            
            # Upload nearest images to Cloudinary and store their URLs
            for path in nearest_image_paths:
                with open(path, "rb") as image_file:
                    nearest_image = BytesIO(image_file.read())
                    nearest_upload_result = cloudinary.uploader.upload(nearest_image)
                    nearest_images.append(nearest_upload_result['secure_url'])

        # Store the Cloudinary URL, the embedding, and the nearest images in MongoDB
        image_collection.insert_one({
            'cloudinary_url': cloudinary_url,
            'embedding': pickle.dumps(feature_vector),
            'nearest_images': nearest_images
        })

        # Return the uploaded image URL and its nearest images
        return jsonify({
            'message': 'Image uploaded successfully',
            'url': cloudinary_url,
            'nearest_images': nearest_images
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

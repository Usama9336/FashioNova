import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors  # Import NearestNeighbors from sklearn
import cv2

# Load precomputed embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained ResNet50 model without the top layer
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add Global Max Pooling to the model
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Corrected image path (make sure the image exists at this path)
img = image.load_img('smaple/shoes.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit model input
preprocessed_img = preprocess_input(expanded_img_array)
`     # Print the shape of the preprocessed image

# Generate feature vector
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)  # Normalize the result

# Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)  # Fit the model with the feature list

# Find nearest neighbors
distances, indices = neighbors.kneighbors([normalized_result])

# Display the nearest images using OpenCV
for file in indices[0][1:6]:
    print(filenames[file])
    temp_img = cv2.imread(filenames[file])  # Read the image file
    resized_img = cv2.resize(temp_img, (256, 256))  # Resize the image
    cv2.imshow('output', resized_img)  # Display the image
    cv2.waitKey(0)  # Wait for a key press to show the next image

# Destroy all windows after displaying
cv2.destroyAllWindows()

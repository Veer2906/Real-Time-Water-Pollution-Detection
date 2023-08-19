import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from sklearn.cluster import KMeans
from tqdm import tqdm  # Import tqdm for the loading bar

# Load the first 20 images
image_folder = 'C:\\Users\\veera\\Desktop\\Veer\\Coding\\Python\\Veritas\\FloW_IMG\\FloW_IMG\\test\\images'  # Replace with the actual path
image_files = os.listdir(image_folder)[:100]

# Initialize a list to store resized images
resized_images = []

# Resize images to the same dimensions and display loading bar
print("Resizing images and performing clustering:")
for img_file in tqdm(image_files):
    img = io.imread(os.path.join(image_folder, img_file))
    resized_img = transform.resize(img, (1000, 500), anti_aliasing=True)
    resized_images.append(resized_img)

# Convert images to grayscale for better clustering (optional)
gray_images = [color.rgb2gray(img) for img in resized_images]
gray_image_vectors = [img.flatten() for img in gray_images]

# Perform K-means clustering
num_clusters = 10 # Number of clusters you want
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(gray_image_vectors)

# Count the frequency of images in each cluster
cluster_counts = np.bincount(cluster_labels)

# Create a bar graph
plt.bar(range(num_clusters), cluster_counts)
plt.xlabel('Cluster Number')
plt.ylabel('Frequency')
plt.title('Frequency of Images in Each Cluster')
plt.xticks(range(num_clusters))
plt.show()

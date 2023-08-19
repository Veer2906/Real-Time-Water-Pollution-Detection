import os
import cv2
import numpy as np
from tqdm import tqdm

def calculate_similarity(img1, img2):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN-based Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Calculate similarity percentage
    similarity = len(good_matches) / len(kp1) * 100.0
    return similarity

if __name__ == "__main__":
    folder_path = "C:\\Users\\veera\\Desktop\\FloW_IMG\\FloW_IMG\\training\\images"
    image_files = os.listdir(folder_path)
    num_images = min(len(image_files), 30)

    # Load the first 150 images and convert them to grayscale
    images = []
    for i in range(num_images):
        image_path = os.path.join(folder_path, image_files[i])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    # Create an empty similarity matrix
    similarity_matrix = np.zeros((num_images, num_images))

    # Calculate similarity for all pairs of the first 150 images with loading bar
    for i in tqdm(range(num_images), desc="Calculating Similarity", unit="image"):
        for j in range(i, num_images):
            similarity_percentage = calculate_similarity(images[i], images[j])
            similarity_matrix[i, j] = similarity_percentage
            similarity_matrix[j, i] = similarity_percentage

    # Calculate overall similarity for the first 150 images
    overall_similarity = np.mean(similarity_matrix)
    print(f"Overall similarity between the first {num_images} images: {overall_similarity:.2f}%")

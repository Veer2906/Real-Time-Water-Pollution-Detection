import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import random
import matplotlib.pyplot as plt

def calculate_ssim(img1, img2):
    common_size = (1280, 720)
    img1_resized = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), common_size)
    img2_resized = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), common_size)
    similarity_index, _ = ssim(img1_resized, img2_resized, full=True)
    return similarity_index

if __name__ == "__main__":
    folder_path = "C:\\Users\\veera\\Desktop\\Veer\\Coding\\Python\\Veritas\\FloW_IMG\\FloW_IMG\\test\\images"
    image_files = os.listdir(folder_path)
    num_images = min(len(image_files), 800)

    selected_images = np.random.choice(image_files, num_images, replace=False)

    images = []

    for image_file in selected_images:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        images.append(img)

    batch_size = 50
    num_batches = num_images // batch_size

    sim_list = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        for i in tqdm(range(start_idx, end_idx), desc=f"Calculating Structural Similarity (Batch {batch_idx + 1})", unit="image"):
            for j in range(i + 1, end_idx):
                similarity_index = calculate_ssim(images[i], images[j])
                sim_list.append(similarity_index)

        plt.hist(sim_list, bins=50)  
        plt.title(f"Histogram of Structural Similarity Index (Batch {batch_idx + 1})")
        plt.xlabel("Structural Similarity Index")
        plt.ylabel("Frequency")

    overall_similarity = np.mean(sim_list)
    print(f"Overall {num_images} images: {overall_similarity:.2f}")
    plt.show()

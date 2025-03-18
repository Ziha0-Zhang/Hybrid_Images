import numpy as np
from PIL import Image
import hybrid

"""
Important: 
Please pay attention to the path of the image file. Please use relative path in the code below to verify the job.

your_project/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── low_frequency.jpg
│   ├── high_frequency.jpg
│   └── hybrid_image.jpg
├── hybrid.py      
└── main.py        

"""

def load_image(path):
    """
    Load an image from the specified path and convert it to a numpy array.
    """
    try:
        img = Image.open(path)
        img = np.array(img) / 255.0
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def save_image(path, img):
    """
    Save the image to the specified path.
    """
    try:
        img = Image.fromarray(np.uint8(img * 255))  # Convert back to uint8 format
        img.save(path)
        print(f"Picture saved to: {path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def main():
    # Picture Path

    img1_path = './images/image1.jpg'
    img2_path = './images/image2.jpg'

    # Read Picture
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Check whether the picture is loaded successfully
    if img1 is None or img2 is None:
        print("Image reading failed, please check the path and file name!")
        return

    # Parameter setting
    sigma = 12  # Adjust this value to control the blurring effect
    size = 72  # Adjust this value to control the kernel size
    # Generate low-frequency and high-frequency pictures
    low_freq = hybrid.low_pass(img1, sigma, size)
    high_freq = hybrid.high_pass(img2, sigma, size)
    # Generate hybrid image
    hybrid_image = 0.65 * low_freq + 0.35 * high_freq

    # Save low-frequency, high-frequency and hybrid pictures

    save_image('./images/low_frequency.jpg', low_freq)
    save_image('./images/high_frequency.jpg', high_freq)
    save_image('./images/hybrid_image.jpg', hybrid_image)


if __name__ == "__main__":
    main()
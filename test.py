import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d
import math
def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    gauss_kernel = np.zeros((height, width))
    h = np.arange(np.floor(-height/2) + 1, np.floor(height/2) + 1)
    w = np.arange(np.floor(-width/2) + 1, np.floor(width/2) + 1)
    
    coef = (1 / (2 * np.pi * sigma**2))

    for x, x1 in enumerate(h):
        for y, y1 in enumerate(w):
            gauss_kernel[x, y] = coef * np.exp(-(x1**2 + y1**2) / (2 * sigma**2))
            
    norm_gauss = gauss_kernel / np.sum(gauss_kernel)

    return norm_gauss

def generate_reference_gaussian_kernel(sigma, height, width):
    """Generates a 2D Gaussian kernel using scipy.signal."""
    h = np.arange(np.floor(-height / 2) + 1, np.floor(height / 2) + 1)
    w = np.arange(np.floor(-width / 2) + 1, np.floor(width / 2) + 1)

    gaussian_h = gaussian_filter1d(np.ones(height), sigma, axis=0, mode='constant')
    gaussian_w = gaussian_filter1d(np.ones(width), sigma, axis=0, mode='constant')

    kernel_1d_h = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(h**2) / (2 * sigma**2))
    kernel_1d_w = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(w**2) / (2 * sigma**2))

    kernel_2d = np.outer(kernel_1d_h, kernel_1d_w)
    kernel_2d_normalized = kernel_2d / np.sum(kernel_2d)
    return kernel_2d_normalized

def compare_kernels(kernel1, kernel2, tolerance=1e-6):
    """Compares two kernels for near equality."""
    if kernel1.shape != kernel2.shape:
        print(f"Shape mismatch: {kernel1.shape} vs {kernel2.shape}")
        return False
    diff = np.abs(kernel1 - kernel2)
    max_diff = np.max(diff)
    if max_diff > tolerance:
        print(f"Max difference: {max_diff:.8f} exceeds tolerance: {tolerance}")
        return False
    return True

if __name__ == "__main__":
    test_cases = [
        (1.0, 3, 3),
        (1.5, 5, 5),
        (2.0, 7, 9),
        (0.5, 9, 7),
        (3.0, 11, 11),
        (1.0, 4, 4), # Even dimensions
        (1.5, 6, 8), # Even dimensions
    ]

    print("Running stress test for gaussian_blur_kernel_2d:")
    for sigma, height, width in test_cases:
        print(f"\nTesting with sigma={sigma}, height={height}, width={width}")
        your_kernel = gaussian_blur_kernel_2d(sigma, height, width)
        reference_kernel = generate_reference_gaussian_kernel(sigma, height, width)

        #print("Your Kernel:")
        #print(your_kernel)
        #print("Reference Kernel (SciPy):")
        #print(reference_kernel)

        if compare_kernels(your_kernel, reference_kernel):
            print("Kernels match (within tolerance).")
        else:
            print("Kernels do NOT match.")
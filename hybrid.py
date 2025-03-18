import numpy as np

def cross_correlation_2d(img, kernel):
    '''
    Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).
    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    u, v = kernel.shape
    new_img = np.zeros(img.shape)
    u_pad = u // 2
    v_pad = v // 2
    if len(img.shape) > 2:
        x, y, colors = img.shape
        padded_img = np.pad(img, ((u_pad, u_pad), (v_pad, v_pad), (0, 0)), mode='constant')
        for i in range(x):
            for j in range(y):
                for color in range(colors):
                    new_img[i, j, color] = np.sum(padded_img[i:i+u, j:j+v, color] * kernel)
    else:
        x, y = img.shape
        # create padded image
        padded_img = np.pad(img, ((u_pad, u_pad), (v_pad, v_pad)), mode='constant')
        for i in range(x):
            for j in range(y):
                new_img[i, j] = np.sum(padded_img[i:i+u, j:j+v] * kernel)
    return new_img

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

def low_pass(img, sigma, size):
    '''Filter the image as if it's filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter suppresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    '''Filter the image as if it's filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)

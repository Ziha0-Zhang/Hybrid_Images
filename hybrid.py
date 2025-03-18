import sys
import numpy as np

import os


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).'''

    u, v = kernel.shape
    new_img = np.zeros_like(img)

    # define padding size
    u_pad = u // 2
    v_pad = v // 2

    # if picture is rgb
    if len(img.shape) > 2:
        x, y, colors = img.shape

        # create padded image
        padded_img = np.pad(img, ((u_pad, u_pad), (v_pad, v_pad), (0, 0)), mode='constant')

        # loop
        for i in range(x):
            for j in range(y):
                for color in range(colors):
                    new_img[i, j, color] = np.sum(padded_img[i:i+u, j:j+v, color] * kernel)

    else:
        x, y = img.shape

        # create padded image
        padded_img = np.pad(img, ((u_pad, u_pad), (v_pad, v_pad)), mode='constant')

        # loop
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
    h_range = np.floor(height / 2)  # The range for the y-coordinate
    w_range = np.floor(width / 2)  # The range for the x-coordinate

    gauss_kernel = np.zeros((height, width))
    coef = (1 / (2 * np.pi * sigma ** 2))
    for x in range(height):
        for y in range(width):
            dx = x - h_range
            dy = y - w_range
            gauss_kernel[x, y] = coef * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))


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
    return cross_correlation_2d(img, gaussian_blur_kernel_2d(sigma, size, size))  # cross_correlation_2d(img, gaussian_blur_kernel_2d(...))


def high_pass(img, sigma, size):
    '''Filter the image as if it's filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return np.clip((img - low_pass(img, sigma, size)),0,1)  # img - ...

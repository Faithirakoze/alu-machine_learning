#!/usr/bin/env python3
"""Module for performing same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    Args:
        images (numpy.ndarray): shape (m, h, w)
        kernel (numpy.ndarray): shape (kh, kw)

    Returns:
        numpy.ndarray: convolved images of shape (m, h, w)
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute padding
    ph = kh // 2
    pw = kw // 2

    # Apply padding
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    # Output array
    output = np.zeros((m, h, w))

    # Only TWO loops (over spatial dimensions)
    for i in range(h):
        for j in range(w):
            current = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(current * kernel, axis=(1, 2))

    return output

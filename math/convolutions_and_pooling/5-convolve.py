#!/usr/bin/env python3
"""Module for performing convolution with multiple kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs convolution on images using multiple kernels

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
        kernels (numpy.ndarray): shape (kh, kw, c, nc)
        padding: 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: convolved images
    """

    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if c != kc:
        raise ValueError("Kernel channels must match image channels")

    # Padding
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    # Output dimensions
    output_h = int((h + 2 * ph - kh) / sh) + 1
    output_w = int((w + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w, nc))

    # THREE loops allowed
    for i in range(output_h):
        for j in range(output_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            current = padded[:, h_start:h_end, w_start:w_end, :]

            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    current * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output

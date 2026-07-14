#!/usr/bin/env python3
"""Module for performing pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
        kernel_shape (tuple): (kh, kw)
        stride (tuple): (sh, sw)
        mode (str): 'max' or 'avg'

    Returns:
        numpy.ndarray: pooled output
    """

    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Compute output dimensions
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c))

    # ONLY TWO LOOPS
    for i in range(out_h):
        for j in range(out_w):

            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            window = images[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output

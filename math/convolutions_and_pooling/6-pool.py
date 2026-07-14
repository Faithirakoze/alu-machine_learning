#!/usr/bin/env python3
import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Padding
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Apply padding
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, nc))

    # ONLY 3 LOOPS
    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                slice_img = images_padded[:, h_start:h_end, w_start:w_end, :]
                kernel = kernels[:, :, :, k]

                output[:, i, j, k] = np.sum(slice_img * kernel, axis=(1, 2, 3))

    return output

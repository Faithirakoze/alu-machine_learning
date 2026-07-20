#!/usr/bin/env python3
"""Calculate the definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """Returns the definiteness of a matrix."""

    # Check type
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check valid shape
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check non-empty
    if matrix.size == 0:
        return None

    # Check symmetry
    if not np.allclose(matrix, matrix.T):
        return None

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Classification
    if np.all(eigenvalues > 0):
        return "Positive definite"

    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    if np.all(eigenvalues < 0):
        return "Negative definite"

    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None

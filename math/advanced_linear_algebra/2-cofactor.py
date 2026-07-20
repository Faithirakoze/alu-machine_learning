#!/usr/bin/env python3
"""Calculate the cofactor matrix of a matrix."""


def determinant(matrix):
    """Calculate determinant recursively."""

    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )

    det = 0
    for j in range(len(matrix)):
        submatrix = [
            [matrix[i][k] for k in range(len(matrix)) if k != j]
            for i in range(1, len(matrix))
        ]
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)

    return det


def minor(matrix):
    """Calculate minor matrix."""

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if n == 1:
        return [[1]]

    minor_matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [
                [matrix[r][c] for c in range(n) if c != j]
                for r in range(n) if r != i
            ]
            row.append(determinant(submatrix))
        minor_matrix.append(row)

    return minor_matrix


def cofactor(matrix):
    """Returns the cofactor matrix of a matrix."""

    # Validate input
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Special case
    if n == 1:
        return [[1]]

    # Get minor matrix
    minor_matrix = minor(matrix)

    # Apply cofactor signs
    cofactor_matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            sign = (-1) ** (i + j)
            row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(row)

    return cofactor_matrix

#!/usr/bin/env python3
"""Calculate the determinant of a matrix."""


def determinant(matrix):
    """Returns the determinant of a matrix."""

    # Handle 0x0 matrix
    if matrix == [[]]:
        return 1

    # Validate: must be list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Validate: must be list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Validate: must be square
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2
    if n == 2:
        return (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )

    # Recursive case
    det = 0
    for j in range(n):
        submatrix = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)

    return det

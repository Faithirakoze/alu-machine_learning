def determinant(matrix):
    # Validate: must be a list of lists
    if (
        not isinstance(matrix, list)
        or len(matrix) == 0
        or not all(isinstance(row, list) for row in matrix)
    ):
        raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix [[]]
    if matrix == [[]]:
        return 1

    n = len(matrix)

    # Validate: must be square
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive cofactor expansion along row 0
    det = 0
    for j in range(n):
        # Build the submatrix (minor) by removing row 0 and column j
        submatrix = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(submatrix)

    return det

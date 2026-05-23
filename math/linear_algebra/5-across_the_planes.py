#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[r][c] + mat2[r][c]
            for c in range(len(mat1[0]))] for r in range(len(mat1))]

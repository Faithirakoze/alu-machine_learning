#!/usr/bin/env python3

def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(mat1[r][k] * mat2[k][c] for k in range(len(mat2)))
             for c in range(len(mat2[0]))]
            for r in range(len(mat1))]

#!/usr/bin/env python3

def matrix_transpose(matrix):
    return [[row[col] for row in matrix] for col in range(len(matrix[0]))]

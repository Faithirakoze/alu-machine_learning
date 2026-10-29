#!/usr/bin/env python3
"""Load a CSV file into a pandas DataFrame."""

import pandas as pd


def from_file(filename, delimiter):
    """Read a delimited file and return its contents as a DataFrame."""
    return pd.read_csv(filename, sep=delimiter)

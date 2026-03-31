"""
utils/data_loader.py
Loads the property catalog from the parquet file into a DataFrame.
"""

import os
import sys
import pandas as pd

synthetic_data = {
    "address": ["123 Main St", "456 Oak Ave", "789 Pine Ln"],
    "month": ["2026-M01", "2026-M02", "2026-M02"],
    "quarter": ["2026-Q1", "2026-Q1", "2026-Q1"],
    "year": ["2026", "2026", "2026"],
    "price": [520000, 475000, 300000],
    "pnl": [10000, -5000, 20000]
}

def load_properties(path: str | None = None, synthetic: bool = False) -> pd.DataFrame:
    """
    Load property catalog from a parquet file.

    Parameters
    ----------
    path : str, optional
        Path to the parquet file. Defaults to DATA_PATH env variable
        or 'data/cortex.parquet'.

    synthetic : bool, optional
        Whether to load synthetic data instead of from a parquet file.

    Returns
    -------
    pd.DataFrame
        Property catalog.

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist at the given path.
    """
    if not synthetic:
        if path is None:
            path = os.getenv("DATA_PATH", "../data/cortex.parquet")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Property data not found at: {path}")

        df = pd.read_parquet(path)

    else:
        df = pd.DataFrame(synthetic_data)
    
    # Normalise address column for consistent matching
    if 'adress' in df.columns:
        df["address"] = df["address"].astype(str).str.strip()

    return df

if __name__ == "__main__":
    # Test loading the data
    try:
        df = load_properties(synthetic=True)
        print("Data loaded successfully. Sample:")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
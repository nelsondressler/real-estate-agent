"""
tools/data_tools.py
Pure-function data access layer over the property DataFrame.
Agents call these functions instead of querying the DataFrame directly.
"""

from typing import Optional
import pandas as pd

def find_property(address: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Case-insensitive partial address match.

    Parameters
    ----------
    address : str
        Address substring to search for.
    df : pd.DataFrame
        Property catalog DataFrame.

    Returns
    -------
    Optional[pd.Series]
        First matching row, or None if not found.
    """
    if 'address' in df.columns:
        mask = df["address"].str.lower().str.contains(address.lower(), na=False)
        results = df[mask]
    elif 'property_name' in df.columns:
        mask = df["property_name"].str.lower().str.contains(address.lower(), na=False)
        results = df[mask]
    else:
        return None
    
    return results.iloc[0] if not results.empty else None


def get_all_properties(
    df: pd.DataFrame,
    year: int | None = None,
    quarter: int | None = None,
    month: int | None = None
) -> pd.DataFrame:
    """Return the full property DataFrame ."""
    filtered_df = df.copy()
    if year is not None:
        filtered_df = filtered_df[filtered_df["year"] == str(year)]
    if quarter is not None:
        filtered_df = filtered_df[filtered_df["quarter"].str.contains(f"Q{quarter}", na=False)]
    if month is not None:
        filtered_df = filtered_df[filtered_df["month"].str.contains(f"M{month:02d}", na=False)]
    
    return filtered_df


def calculate_pnl(row: pd.Series) -> float:
    """
    Compute Profit & Loss for a single property.

    Formula
    -------
    P&L = (profit - loss)

    Parameters
    ----------
    row : pd.Series
        A single property row from the DataFrame.

    Returns
    -------
    float
        Calculated P&L value.
    """
    if 'pnl' in row:
        result = row["pnl"]
    elif 'profit' in row:
        result = row["profit"]
    else:
        result = 0.0
    
    return result

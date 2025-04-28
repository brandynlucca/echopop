from typing import Optional, Tuple
import numpy as np
import pandas as pd


# Include:
# - other functions needed to create mesh

def create_mesh(
    df_transect: pd.DataFrame,
    df_mesh_template: pd.DataFrame,
    crop: bool,
    crop_method: str,
) -> pd.DataFrame:
    """
    Create mesh based on a mesh template and the transect lat/lon.
    """
    pass


def stratify_mesh(df_mesh: pd.DataFrame) -> pd.DataFrame:
    """
    Add "stratum" column to the mesh dataframe.
    """
    pass


def transform_geometry(
    df_in: pd.DataFrame,
    df_ref_grid: pd.DataFrame,  # why does this need to be a geodataframe?
    longitude_reference: float,
    longitude_offset: float,
    latitude_offset: float,
    delta_longitude: Optional[float] = None,
    delta_latitude: Optional[float] = None,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Transform lat/lon to x/y with specific references.
    """
    pass

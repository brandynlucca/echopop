from typing import Union, Dict, Tuple
from pathlib import Path
import pandas as pd


# Break up load_data() and read_validated_data() into 3 separate functions to load biological, stratification, and kriging data
# Don't worry about validating input files for now

# TODO: vario_krig_para is currently not used, right? If so, remove any related code


# TODO: combine in content of preprocess_biodata()
def load_biological_data(root_path: Union[str, Path], file_path_dict: Dict) -> Dict[pd.DataFrame]:
    """
    Load biological data from CSV

    Parameters
    ----------
    root_path : str or Path
        Path to the biological data file
    file_path_dict : dict
        Dictionary of paths to individual biological data files

    Returns
    -------
    A dictionary of dataframes containing biological data
    """
    df_bio_dict: Dict
    return df_bio_dict


# TODO: combine in the current preprocess_spatial()
def load_stratification(root_path: Union[str, Path], file_path_dict: Dict) -> Dict[pd.DataFrame]:
    """
    Load stratification schemes from CSV

    Parameters
    ----------
    root_path : str or Path
        Path to stratification CSV
    file_path_dict : dict
        Dictionary of paths to individual stratification files

    Returns
    -------
    A dictionary of dataframes containing bio_strate and geo_strata info
    """

    # `bio_strata` is the current `strata`

    # In addition to the original operations in preprocess_spatial()
    # also do stratum renaming for inpfc originall in transect.py::save_transect_coordinates()
    # if stratum_def == "inpfc":
    #     stratum_rename = "stratum_num"
    # else:
    #     stratum_rename = stratum_col

    df_strata_dict: Dict
    return df_strata_dict


# same as the current preprocess_biology_spatial()
def join_biological_stratification(
    df_bio_dict: Dict[pd.DataFrame], df_strata_dict: Dict[pd.DataFrame]
) -> Dict[pd.DataFrame]:
    return df_bio_dict


# same as the current preprocess_acoustic_spatial()
def join_acoustic_stratification(
    df_nasc: pd.DataFrame, df_strata_dict: Dict[pd.DataFrame]
) -> pd.DataFrame:
    return df_nasc


# same as the current preprocess_acoustic_biology_spatial()
def join_acoustic_all(
    df_nasc: pd.DataFrame,
    df_bio_dict: Dict[pd.DataFrame],
    df_strata_dict: Dict[pd.DataFrame],
    species_code
) -> pd.DataFrame:
    return df_nasc


# NOTE: combine content of preprocess_statistics()
def load_kriging_templates(
    root_path: Union[str, Path], file_path_dict: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load kriging input

    Parameters
    ----------
    root_path : str or Path
        Path to CSV
    file_path_dict : dict
        Dictionary of paths to individual kriging and isobath files

    Returns
    -------
    A dictionary of dataframes containing kriging and isobath info
    """
    df_mesh: pd.DataFrame
    df_isobath: pd.DataFrame
    return df_mesh, df_isobath


# separate out components that are parameters
def load_kriging_variogram_params(root_path: Union[str, Path], file_path_dict: Dict) -> Tuple[Dict, Dict]:
    kriging_params_dict: dict
    variogram_params_dict: dict
    return kriging_params_dict, variogram_params_dict

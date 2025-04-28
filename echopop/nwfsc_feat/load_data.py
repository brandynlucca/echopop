from typing import Union, Dict
from pathlib import Path
import pandas as pd


# Break up load_data() and read_validated_data() into 3 separate functions to load biological, stratification, and kriging data
# Don't worry about validating input files for now

# TODO: vario_krig_para is currently not used, right? If so, remove any related code


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
    df_strata_dict: Dict
    return df_strata_dict


def load_kriging_params(root_path: Union[str, Path], file_path_dict: Dict) -> Dict[pd.DataFrame]:
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
    df_kriging_dict: Dict
    return df_kriging_dict


# same as the current preprocess_biodata()
def clean_biological_data(df_bio_dict: Dict[pd.DataFrame]) -> Dict[pd.DataFrame]:
    return df_bio_dict


# same as the current preprocess_statistics()
def update_kriging(
    df_kriging_dict: Dict[pd.DataFrame],
    kriging_params: Dict,
) -> Dict[pd.DataFrame]:
    return df_kriging_dict


# same as the current preprocess_spatial()
def clean_stratification(df_strata_dict: Dict[pd.DataFrame]) -> Dict[pd.DataFrame]:

    # In addition to the original operations
    # also do stratum renaming for inpfc originall in transect.py::save_transect_coordinates()
    # if stratum_def == "inpfc":
    #     stratum_rename = "stratum_num"
    # else:
    #     stratum_rename = stratum_col


    return df_strata_dict


# same as the current preprocess_acoustic_spatial()
def join_acoustic_stratification(
    df_acoustic_dict: Dict[pd.DataFrame], df_strata_dict: Dict[pd.DataFrame]
) -> Dict[pd.DataFrame]:
    return df_acoustic_dict

# same as the current preprocess_biology_spatial()
def join_biological_stratification(
    df_bio_dict: Dict[pd.DataFrame], df_strata_dict: Dict[pd.DataFrame]
) -> Dict[pd.DataFrame]:
    return df_bio_dict


# same as the current preprocess_acoustic_biology_spatial()
def join_acoustic_all(
    df_acoustic_dict: Dict[pd.DataFrame],
    df_bio_dict: Dict[pd.DataFrame],
    df_strata_dict: Dict[pd.DataFrame],
    species_code
) -> Dict[pd.DataFrame]:
    return df_acoustic_dict

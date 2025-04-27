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



# Scripts to execute what's in Survey.load_survey_data()
# All *_dict below are a subdict from the original config yaml

root_path = "WHERE_ALL_DATA_IS"
species_code = "SPECIES_CODE"
df_acoustic_dict: Dict[pd.DataFrame]  # Load extract nasc data

bio_path_dict: Dict  # the "biological" section of year_config.yml
strata_path_dict: Dict  # the "stratification" section of year_config.yml
kriging_path_dict: Dict  # the "kriging" section of year_config.yml
kriging_param_dict: Dict  # the "kriging_parameters" section of init_config.yml

df_bio_dict: Dict[pd.DataFrame] = load_biological_data(root_path, file_path_dict=bio_path_dict)
df_bio_dict = clean_biological_data(df_bio_dict)

df_strata_dict: Dict[pd.DataFrame] = load_stratification(root_path, file_path_dict=strata_path_dict)
df_strata_dict = clean_stratification(df_strata_dict)

# TODO: Consider combining update_kriging() into load_kriging_params since it's simpler
df_kriging_dict: Dict[pd.DataFrame] = load_kriging_params(root_path, file_path_dict=kriging_path_dict)
df_kriging_dict = update_kriging(df_kriging_dict, kriging_params=kriging_param_dict)

# Consolidate all input data into df_acoustic_dict
df_bio_dict = join_biological_stratification(df_bio_dict, df_strata_dict)
df_acoustic_dict = join_acoustic_stratification(df_acoustic_dict, df_strata_dict)
df_acoustic_dict = join_acoustic_all(df_acoustic_dict, df_bio_dict, df_strata_dict, species_code)

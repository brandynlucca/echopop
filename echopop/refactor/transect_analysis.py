from typing import Union, Literal, Dict
import numpy as np
import pandas as pd


# same as the current save_transect_coordinates but with explicit inputs
def get_transect_coordinates(
    df_nasc: pd.DataFrame,
    age_type: Literal["no_age1", "all_ages"],
    stratum_type: Literal["ks", "inpfc"],
) -> pd.DataFrame:
    # NOTE: inpfc stratum renaming should be done in load_data.py::clean_stratification()
    pass


def ts_length_regression(
    length: Union[np.ndarray, float], slope: float, intercept: float
) -> np.ndarray:
    pass


# same as the current aggregate_sigma_bs but with explicit inputs
def get_stratified_sigma_bs(
    df_bio_dict: Dict[pd.DataFrame],  # df_bio_dict from end of load_data.py
    ts_length_regression_dict: dict,  # only 1 species so do not need species code and related checking
    stratum_type: Literal["ks", "inpfc"],  # only do 1 type at each call
) -> Dict[pd.DataFrame]:
    
    # Organize bio data
    # -- meld the specimen and length dataframes together for downstream calculations
    # -- regroup the length-specific length counts

    # Merge bio and regression dfs
    # -- create DataFrame containing all necessary regression coefficients
    # -- merge with the biological data
    # -- calculate predicted TS from the length values
    # -- convert TS to the linear domain ('sigma_bs')

    # Calculate mean sigma_bs for all hauls and the specified stratum type
    # -- impute sigma_bs values, if necessary, for missing strata
    # -- calculate mean TS for all hauls, KS-strata, and INPFC strata

    df_sigma_bs_haul: pd.DataFrame
    df_sigma_bs_stratum: pd.DataFrame

    return {"df_haul_mean": df_sigma_bs_haul, "df_strata_mean": df_sigma_bs_stratum}


# from the current fit_length_weight_relationship()
def get_fitted_weight(
    df_specimen: pd.DataFrame,  # from df_bio_dict from load_data.py
    df_length_bins: pd.DataFrame,
    species: int,
) -> pd.DataFrame:
    # NOTE: I think you can remove the fitted relationship from the output unless it is used elsewhere in the codebase
    pass


# from the current quantize_number_counts()
def get_fish_count(
    df_specimen: pd.DataFrame,  # from df_bio_dict from load_data.py
    df_length: pd.DataFrame,  # from df_bio_dict from load_data.py
    aged: bool,
    sexed: bool
) -> pd.DataFrame:
    # NOTE: make the function only do 1 thing for each call, not all of it, so output is simpler
    pass


# from the current number_proportions()
def get_number_proportion(
    df_aged, df_unaged, df_weight,  # only include the necessary dfs -- I lost track
    proportion_type: Literal["age", "sex", "unaged_length", "unaged_length"]
) -> pd.DataFrame:
    # NOTE: Do 1 species in one call
    pass


# NOTE: I lost track what's going on in the function
# let's talk through the last 3 functions in process_transect_data()

def quantize_weights():
    # NOTE: I cannot figure out what's going on in here
    # but I think the output is probably better represented as a N-D array
    # you can use xarray.Dataset to organize this type of data
    # it will also help with the slicing you have to do in the downstream weight_proportions
    pass


# from the current fit_length_weights()
def get_stratum_averaged_weight():
    pass


# from the current weight_proportions()
def get_weight_proportion():
    pass
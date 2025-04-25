from typing import Literal
import pandas as pd


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
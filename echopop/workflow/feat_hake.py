from typing import Dict
import pandas as pd

from echopop.nwfsc_feat import load
from nwfsc_feat import ingest_echoview_nasc


# ===========================================
# Organize NASC file
nasc_path = "SOME_PATH"
nasc_filename_pattern = "SOME_PATTERN"
region_class_mapping = {}  # pattern-label mapping under transect_region_mapping/parts

df_merged = ingest_echoview_nasc.merge_echoview_nasc(
    nasc_path, nasc_filename_pattern)
df_transect_region_key = ingest_echoview_nasc.construct_transect_region_key(
    df_merged, region_class_mapping)

# Use df.to_csv to save df_transect_region_key, in place of the specialized transect_region_key file
# Keep read_transect_region_file and make sure its output is the same as construct_transect_region_key


# Age-1+
df_nasc_all_ages = ingest_echoview_nasc.consolidate_echoview_nasc(
    df_merged,
    region_names=["Age-1 Hake", "Age-1 Hake Mix", "Hake", "Hake Mix"]
)

# Age-2+ (no age 1)
df_nasc_no_age1 = ingest_echoview_nasc.consolidate_echoview_nasc(
    df_merged,
    region_names=["Hake", "Hake Mix"]
)

# Use df.to_csv to save df_nasc_all_ages and df_nasc_no_age1 if needed

# Use regular pd.read_csv to read df_nasc_*, effectively break up the current load_data()
# -- there is no need to have a one-size-fits-all load_data function
# -- you can validate the biological, stratification, and NASC data
# -- but just read them in is fine: these are all files under our control




# ===========================================
# Execute what's in Survey.load_survey_data()
# All *_dict below are a subdict from the original config yaml

root_path = "WHERE_ALL_DATA_IS"
species_code = "SPECIES_CODE"
df_acoustic_dict: Dict[pd.DataFrame]  # Load extract nasc data

bio_path_dict: dict  # the "biological" section of year_config.yml
strata_path_dict: dict  # the "stratification" section of year_config.yml
kriging_path_dict: dict  # the "kriging" section of year_config.yml
kriging_param_dict: dict  # the "kriging_parameters" section of init_config.yml

df_bio_dict: Dict[pd.DataFrame] = load.load_biological_data(root_path, file_path_dict=bio_path_dict)
df_bio_dict = load.clean_biological_data(df_bio_dict)

df_strata_dict: Dict[pd.DataFrame] = load.load_stratification(root_path, file_path_dict=strata_path_dict)
df_strata_dict = load.clean_stratification(df_strata_dict)

# TODO: Consider combining update_kriging() into load_kriging_params since it's simpler
df_kriging_dict: Dict[pd.DataFrame] = load.load_kriging_params(root_path, file_path_dict=kriging_path_dict)
df_kriging_dict = load.update_kriging(df_kriging_dict, kriging_params=kriging_param_dict)

# Consolidate all input data into df_acoustic_dict
df_bio_dict = load.join_biological_stratification(df_bio_dict, df_strata_dict)
df_acoustic_dict = load.join_acoustic_stratification(df_acoustic_dict, df_strata_dict)
df_acoustic_dict = load.join_acoustic_all(df_acoustic_dict, df_bio_dict, df_strata_dict, species_code)

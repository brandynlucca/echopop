from typing import Dict, Union, Tuple
from pathlib import Path
import pandas as pd

from echopop.nwfsc_feat import ingest_echoview_nasc, load_data


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

root_path = "WHERE_ALL_DATA_ARE"
species_code = "SPECIES_CODE"
df_nasc_no_age1: pd.DataFrame  # extracted nasc data from above

bio_path_dict: dict  # the "biological" section of year_config.yml
strata_path_dict: dict  # the "stratification" section of year_config.yml
kriging_path_dict: dict  # the "kriging" section of year_config.yml
                         # combined with the "kriging" section of init_config.yml

df_bio_dict: Dict[pd.DataFrame] = load_data.load_biological_data(root_path, file_path_dict=bio_path_dict)

df_strata_dict: Dict[pd.DataFrame] = load_data.load_stratification(root_path, file_path_dict=strata_path_dict)

# Consolidate all input data into df_acoustic_dict
df_bio_dict = load_data.join_biological_stratification(df_bio_dict, df_strata_dict)
df_nasc_no_age1 = load_data.join_acoustic_stratification(df_nasc_no_age1, df_strata_dict)
df_nasc_no_age1 = load_data.join_acoustic_all(df_nasc_no_age1, df_bio_dict, df_strata_dict, species_code)




# ===========================================
# Load kriging-related params and templates
df_mesh_template, df_isobath = load_data.load_kriging_templates(root_path, kriging_path_dict)
kriging_param_dict, variogram_params_dict = load_data.load_kriging_variogram_params(root_path, kriging_path_dict)

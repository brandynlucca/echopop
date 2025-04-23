from echopop.survey import Survey
from echopop.utils.validate_dict import CONFIG_DATA_MODEL, CONFIG_INIT_MODEL, BiologicalFiles, XLSXFile
from pathlib import Path
from echopop.utils import load as el, load_nasc as eln, message as em
import copy
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import yaml

from echopop.core import BIODATA_HAUL_MAP, DATA_STRUCTURE, LAYER_NAME_MAP, NAME_CONFIG
from echopop.utils.data_structure_utils import map_imported_datasets
from echopop.utils.validate_df import DATASET_DF_MODEL
from echopop.utils.validate_dict import CONFIG_DATA_MODEL, CONFIG_INIT_MODEL

survey = Survey(
    "./config_files/initialization_config.yml",
    "./config_files/survey_year_2019_single_biodata_config.yml"
)

self = survey
new_datasets = ["biological", "kriging", "stratification"]
# ---- Load in the new data
input_dict, configuration_dict, dataset_type = self.input, self.config, new_datasets

# Re-initialize the input keys, if needed
# ---- Get name of the proposed input dictionary keys
input_keys = [
    (
        LAYER_NAME_MAP[key]["superlayer"][0]
        if LAYER_NAME_MAP[key]["superlayer"]
        else LAYER_NAME_MAP[key]["name"]
    )
    # for key in CONFIG_MAP.keys()
    for key in DATASET_DF_MODEL.keys()
    if key in list(dataset_type)
]
# ---- Map the complete datasets
imported_data = map_imported_datasets(input_dict)
# ---- Re-initialize if data already loaded to avoid duplication issues
if set(input_keys).issubset(imported_data):
    # ---- Reset the relevant keys
    input_dict.update({key: copy.deepcopy(DATA_STRUCTURE["input"][key]) for key in input_keys})

# Check whether data files defined from the configuration file exists
# ---- Generate flat JSON table comprising all configuration parameter names
flat_configuration_table = pd.json_normalize(configuration_dict).filter(regex="filename")

# Coerce `dataset_type` into List[str], if needed
dataset_type = [dataset_type] if isinstance(dataset_type, str) else dataset_type

# Get the subset table if specific `dataset_type` is defined
if dataset_type:
    # ---- Get the outermost dictionary keys
    outer_keys = flat_configuration_table.columns.str.split(".").str[0]
    # ---- Get the associated column names
    matching_columns = flat_configuration_table.columns[outer_keys.isin(dataset_type)]
    # ---- Filter the columns
    flat_configuration_table = flat_configuration_table.filter(matching_columns)
# ---- Default to `CONFIG_MAP` keys otherwise
else:
    # dataset_type = list(CONFIG_MAP.keys())
    dataset_type = list(DATASET_DF_MODEL.keys())
# ---- Parse the flattened configuration table to identify data file names and paths
parsed_filenames = flat_configuration_table.values.flatten()
# ---- Evaluate whether either file is missing
data_existence = [
    (Path(configuration_dict["data_root_dir"]) / file).exists() for file in parsed_filenames
]

# Assign the existence status to each configuration file for error evaluation
# ---- Error evaluation and print message (if applicable)
if not all(data_existence):
    missing_data = parsed_filenames[~np.array(data_existence)]
    raise FileNotFoundError(f"The following data files do not exist: {missing_data}")

# Get the applicable `CONFIG_MAP` keys for the defined datasets
expected_datasets = set(DATASET_DF_MODEL.keys()).intersection(dataset_type)

# Define root data directory
data_root_directory = Path(configuration_dict["data_root_dir"])

# Data validation and import
# ---- Iterate through known datasets and datalayers
# for dataset in list(expected_datasets):
dataset = "biological"

# for datalayer in [*configuration_dict[dataset].keys()]:
datalayer = "length"

# Define validation settings from CONFIG_MAP
validation_settings = DATASET_DF_MODEL[dataset][datalayer]

# Define configuration settings w/ file + sheet names
config_settings = configuration_dict[dataset][datalayer]

# Create reference index of the dictionary path
config_map = [dataset, datalayer]

# Create list of region id's associated with biodata (if applicable)
if dataset == "biological":
    regions = configuration_dict[dataset][datalayer].keys()
else:
    regions = [None]  # Use None for non-biological datasets

# Create the file and sheetnames for the associated datasets
for region_id in regions:
    if region_id is not None:
        # ---- Biological data
        file_name = data_root_directory / config_settings[region_id]["filename"]
        sheet_name = config_settings[region_id]["sheetname"]
        # ---- Update `config_map` to include region_id
        config_map = [dataset, datalayer, region_id]
    else:
        # ---- All other datasets
        file_name = data_root_directory / config_settings["filename"]
        sheet_name = config_settings["sheetname"]

    # Ensure sheet_name is a list (to handle cases with multiple lists)
    sheet_name = [sheet_name] if isinstance(sheet_name, str) else sheet_name

    # Validate data for each sheet
    for sheets in sheet_name:
        read_validated_data(
            input_dict,
            configuration_dict,
            file_name,
            sheets,
            config_map,
            validation_settings,
        )

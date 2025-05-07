survey = Survey(init_config_path, survey_year_config_path)

self = survey
input_dict = self.input
configuration_dict = self.config
dataset_type = "NASC"
import copy
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import yaml

from echopop.core import BIODATA_HAUL_MAP, DATA_STRUCTURE, LAYER_NAME_MAP, NAME_CONFIG
from echopop.utils.data_structure_utils import map_imported_datasets
from echopop.utils.validate_df import DATASET_DF_MODEL, AcousticData
from echopop.utils.validate_dict import CONFIG_DATA_MODEL, CONFIG_INIT_MODEL
from echopop.utils.load import map_imported_datasets, read_validated_data
import numpy as np

survey_year = survey.config["survey_year"]
survey.config.keys()
afsc_conversion_config = survey.config["AFSC_nasc_conversion"]
transect_interval_distance = afsc_conversion_config["transect_interval_distance"]
transect_spacing = afsc_conversion_config["max_transect_spacing"]
haul_default = afsc_conversion_config["default_haul_id"]
region_id_default = afsc_conversion_config["default_region_id"]
input_files = afsc_conversion_config["input_directory"]
root = survey.config["data_root_dir"]
filename_template = afsc_conversion_config["save_file_template"].replace("{DATA_ROOT_DIR}", root)
export_sheetname = afsc_conversion_config["save_file_sheetname"]
nasc_groups = survey.config["NASC"].keys()
grp = "no_age1"
verbose = True

for grp in nasc_groups:
    # ---- Initialize savefile template
    save_file_template = filename_template
    # ---- Prepare filename
    file_name = Path(root) / survey.config["NASC"][grp]["filename"]
    # ---- Prepare sheetname
    sheet_name = survey.config["NASC"][grp]["sheetname"]
    # ---- Read in the file
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    # ---- Force the column names to be lower case
    df.columns = df.columns.str.lower()
    # ---- Apply column name mapping
    df.rename(columns=NAME_CONFIG, inplace=True)
    # ---- Create distance intervals
    df.rename(columns={"distance": "distance_s"}, inplace=True)
    # ---- End of interval
    df["distance_e"] = df["distance_s"] + transect_interval_distance
    # ---- Create placeholder for region ID
    df["region_id"] = region_id_default
    # ---- Fill in empty transect spacings
    df.loc[
        np.where(np.isnan(df["transect_spacing"]))[0], 
        "transect_spacing"
    ] = transect_spacing
    # ----- Create placeholder for haul number
    df["haul_num"] = haul_default
    # ---- Validate the remaining dataframe
    df_validated = AcousticData.validate_df(df)
    # ---- Update survey year
    save_file_template = save_file_template.replace("{YEAR}", str(survey_year))
    # ---- Update export group
    save_file_template = save_file_template.replace("{GROUP}", grp)
    # ---- Save xlsx file
    df_validated.to_excel(
        excel_writer=save_file_template,
        sheet_name=export_sheetname,
        index=False,
    )
    # ---- Print out message
    if verbose:
        print(
            f"Updated NASC export file for group '{grp}' saved at "
            f"'{save_file_template}'."
        )
        
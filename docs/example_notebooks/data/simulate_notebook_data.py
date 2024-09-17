import pandas as pd
import numpy as np
import math
import os 
from pathlib import Path
import matplotlib.pyplot as plt

# Get the current working directory where the notebook is running
notebook_dir = os.getcwd()

# Navigate to the example_data folder
data_dir = os.path.join(notebook_dir, "docs/example_notebooks/data")

def create_equally_spaced_coordinates(lat1, lon1, lat2, lon2, distance):
    """
    Creates an array of equally spaced coordinate pairs between two given points.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point.
        lat2, lon2: Latitude and longitude of the second point.
        distance: Desired distance between coordinate pairs in nautical miles.
        
    Returns:
        A list of coordinate pairs.
    """
    
    # Convert latitude and longitude to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate the distance between the two points in meters
    earth_radius_meters = 6371000.0
    if abs(lat1_rad - lat2_rad) < 1e-6:  # Check if latitude difference is negligible
        d = math.cos(lat1_rad) * earth_radius_meters * abs(lon2_rad - lon1_rad)
        num_pairs = int(d / (distance * 1852)) + 1
        coordinates = []
        for i in range(num_pairs):
            lon = lon1_rad + i * (lon2_rad - lon1_rad) / (num_pairs - 1)
            coordinates.append((math.degrees(lat1_rad), math.degrees(lon)))                    
    else:
        d = math.acos(math.sin(lat1_rad) * math.sin(lat2_rad) + math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)) * earth_radius_meters
        # Calculate the number of coordinate pairs needed
        num_pairs = int(d / (distance * 1852)) + 1  # 1 nautical mile is 1852 meters
    
        # Create an array of equally spaced coordinate pairs
        coordinates = []
        for i in range(num_pairs):
            t = i / (num_pairs - 1)
            lat = math.asin(math.sin(lat1_rad) * math.cos(t * d / earth_radius_meters) + math.cos(lat1_rad) * math.sin(t * d / earth_radius_meters) * math.cos(lon2_rad - lon1_rad))
            lon = lon1_rad + math.atan2(math.sin(t * d / earth_radius_meters) * math.sin(lon2_rad - lon1_rad), math.cos(lat1_rad) * math.cos(t * d / earth_radius_meters) - math.sin(lat1_rad) * math.sin(lat2_rad) * math.cos(lon2_rad - lon1_rad))
            coordinates.append((math.degrees(lat), math.degrees(lon)))
        
    return coordinates

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on a sphere using the Haversine formula.

    Args:
        lat1, 
 lon1: Latitude and longitude of the first point (in radians).
        lat2, lon2: Latitude and longitude of the second point (in radians).

    Returns:
        The distance between the two points in meters.
    """

    earth_radius_meters = 6371000.0  # Earth's radius in meters
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius_meters * c
    return distance


params = pd.DataFrame({
    "transect_num": [1, 2, 3, 4, 5, 6],
    "region_id": [999, 999, 999, 999, 999, 999],
    "lon1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "lon2": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "lat1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lat2": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "transect_spacing": np.repeat([25.0], 6),
    "layer_depth_mean": 999.0,
    "layer_height": 999.0,
    "bottom_depth": 10e3,
    "haul_num": [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10], [9, 10, 11, 12], [11, 12, 13, 14]],
    "group": "the_deep",
})

distance_increment = 1  # nautical miles

transect_df = pd.DataFrame({})
for index, row in params.iterrows():
    lat1, lon1 = row["lat1"], row["lon1"]
    lat2, lon2 = row["lat2"], row["lon2"]
    coordinates = create_equally_spaced_coordinates(lat1, lon1, lat2, lon2, distance_increment)
    # Calculate vessel logs
    dist = [0.0]
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i-1]
        lat2, lon2 = coordinates[i]
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dist.append(haversine_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad) / 1852.0)
    # Convert to start/end
    vl_start = []
    vl_end = []
    cdist = np.cumsum(dist)
    for i in range(1, len(cdist)):
        vl_start.append(cdist[i-1])
        vl_end.append(cdist[i])
    # Calculate center coordinates
    lat = []
    lon = []
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i-1]
        lat2, lon2 = coordinates[i]
        lat.append((lat2+lat1)/2.0)
        lon.append((lon2+lon1)/2.0)
    # Randomly generate populated NASC
    vals = np.random.binomial(1, 0.5, size=len(coordinates) - 1) # presence/absence
    # ---- Simulate haul numbers
    haul_num = np.where(vals == 0.0, 0, 1)
    # ---- Assign haul numbers
    # -------- Length of non-zeros
    haul_len = len(haul_num[haul_num] > 0)
    if haul_len > 0:
        # --- Get row hauls
        row_hauls = row["haul_num"]
        valid_coords = pd.Series(haul_num).index[haul_num > 0].to_numpy()
        delta_valid_coords = np.where(np.diff(valid_coords) > 1)[0]
        idx_groups = np.digitize(valid_coords, valid_coords[delta_valid_coords], right=True)
        if len(idx_groups) <= len(row_hauls):
            haul_num[valid_coords] = np.array(row_hauls)[idx_groups]
        else:
            idx_groups[idx_groups > (len(row_hauls) - 1)] = len(row_hauls) - 1
            haul_num[valid_coords] = np.array(row_hauls)[idx_groups]
    # ------- Simulate NASC
    nasc_sim = np.random.lognormal(10, 1.0, size=haul_len)
    # ------- Assign
    nasc = np.where(haul_num > 0, nasc_sim, 0.0)       
    # Create DataFrame
    transect_data = pd.DataFrame({
        "transect_num": row["transect_num"],
        "region_id": row["region_id"],
        "vessel_log_start": vl_start,
        "vessel_log_end": vl_end,
        "latitude": lat,
        "longitude": lon,
        "transect_spacing": row["transect_spacing"],
        "layer_depth_mean": row["layer_depth_mean"],
        "layer_height": row["layer_height"],
        "bottom_depth": row["bottom_depth"],
        "NASC": nasc,
        "haul_num": haul_num,
        "group": row["group"],
    })
    # CONCAT
    transect_df = pd.concat([transect_df, transect_data])

# pd.read_excel(Path(data_dir) / "NASC_table_the_deep.xlsx", sheet_name="Sheet1")

# WRITE EXCEL : NASC
# transect_df.to_excel(
#     excel_writer=Path(data_dir) / "NASC_table_the_deep.xlsx",
#     sheet_name="Sheet1",
#     index=False
# )

####################################################################################################
# CATCH DATA
# get valid hauls from transect data
bio_hauls = transect_df["haul_num"].unique()
# ---- Add spp id
spp_id = 999999
# ---- Iterate through
catch_df = pd.DataFrame({
    "haul_num": bio_hauls[bio_hauls > 0],
    "species_id": spp_id,
    "haul_weight": np.random.lognormal(6, 1.0, size=len(bio_hauls[bio_hauls > 0]))
})

# WRITE EXCEL : CATCH_DF
catch_df.to_excel(
    excel_writer=Path(data_dir) / "biodata_catch.xlsx",
    sheet_name="biodata_catch",
    index=False
)
####################################################################################################
# LENGTH DATA
# ---- spoof length distribution
length_distrib = np.linspace(50, 200, 26)
# ---- full choice
length_full = np.arange(50.0, 201.0)
# ---- Iterate through
length_df = pd.DataFrame({})
for i in bio_hauls[bio_hauls > 0]:
    # randomly generate number of values
    num_bins = np.random.randint(1, int(np.ceil(len(length_full) / 4)) + 1)
    lengths = np.random.choice(length_full, size=num_bins, replace=False)
    length_counts = np.random.poisson(30, size=num_bins)
    # ---- Assign sex
    sex_sim = np.random.choice([1, 2, 3], size=num_bins, p=np.array([0.475, 0.475, 0.050]))
    # ----
    length_df = pd.concat([length_df,
                           pd.DataFrame({
                                "haul_num": i,
                                "species_id": spp_id,
                                "sex": sex_sim,
                                "length": lengths,
                                "length_count": length_counts,
                            })])

# WRITE EXCEL : LENGTH_DF
length_df.to_excel(
    excel_writer=Path(data_dir) / "biodata_length.xlsx",
    sheet_name="biodata_length",
    index=False
)
####################################################################################################
# SPECIMEN DATA
# ----
def symmetric_distribution(age_bins, mean, sigma):
    # Calculate the probability density for each bin
    densities = np.exp(-0.5 * ((age_bins - mean) / sigma) ** 2)
    
    # Normalize to make the total sum of densities equal to 1
    densities /= np.sum(densities)
    
    return densities
# --- Assign age distribution
age_bins = np.linspace(1, 20, 20).astype(int)
# ---- Synthesize weighted distribution
mean = np.mean(age_bins)
sigma = 2  # You can adjust the standard deviation
age_p = symmetric_distribution(age_bins, mean, sigma)
# plt.figure(figsize=(10, 6))
# plt.bar(age_bins, age_p, width=0.8, color='skyblue', edgecolor='black')
# plt.xlabel('Age')
# plt.ylabel('Probability Density')
# plt.title('Symmetric Distribution of Age Bins')
# plt.show()
# ---- Iterate through
specimen_df = pd.DataFrame({})
for i in bio_hauls[bio_hauls > 0]:
    # randomly generate number of values
    num_specimens = np.random.randint(10, 100)
    # draw lengths
    lengths = np.random.choice(length_full, size=num_specimens, replace=True)
    # assign sex
    sex_sim = np.random.choice([1, 2, 3], size=num_specimens, p=np.array([0.475, 0.475, 0.050]))
    # ---- Assign age
    age_sim = np.random.choice(age_bins, size=num_specimens, p=age_p)
    # ---- assign weigbht
    wgt_sim = np.random.gamma(2, 1, size=num_specimens)    
    # ---- CONCAT
    specimen_df = pd.concat([specimen_df,
                             pd.DataFrame({
                                 "haul_num": i,
                                 "species_id": spp_id,
                                 "sex": sex_sim,
                                 "length": lengths,
                                 "weight": wgt_sim,
                                 "age": age_sim,
                            })])

# WRITE EXCEL : LENGTH_DF
specimen_df.to_excel(
    excel_writer=Path(data_dir) / "biodata_specimen.xlsx",
    sheet_name="biodata_specimen",
    index=False
)
####################################################################################################
# HAUL-TO-TRANSECT
haul_to_transect_df = (
    transect_df.filter(["haul_num", "transect_num"])
    .drop_duplicates().sort_values(["haul_num"]).loc[lambda x: x.haul_num > 0]
)

# WRITE EXCEL : LENGTH_DF
haul_to_transect_df.to_excel(
    excel_writer=Path(data_dir) / "haul_to_transect_mapping.xlsx",
    sheet_name="Sheet1",
    index=False
)

####################################################################################################
# STRATIFICATION [INPFC + KS]
# ---- num values
num_hauls = len(bio_hauls[bio_hauls > 0])
# ----- KS
ks_strata = pd.DataFrame({
    "stratum_num": np.random.randint(1, 6, size=num_hauls),
    "haul_num": bio_hauls[bio_hauls > 0],
    "fraction_hake": 1.0,
})

# WRITE
ks_strata.to_excel(
    excel_writer=Path(data_dir) / "strata.xlsx",
    sheet_name="Base KS",
    index=False
)

# WRITE EXCEL : NASC
transect_mrg = transect_df.merge(ks_strata, how="left")
transect_mrg["stratum_num"] = np.where(
    np.isnan(transect_mrg["stratum_num"]),
    1,
    transect_mrg["stratum_num"]
).astype(int)
transect_mrg["fraction_hake"] = transect_mrg["fraction_hake"].fillna(0.0)

transect_mrg.to_excel(
    excel_writer=Path(data_dir) / "NASC_table_the_deep.xlsx",
    sheet_name="Sheet1",
    index=False
)

# ----- INPFC
# ------ break up transect latitudes into 3 parts
lat_bins = np.concatenate([
    np.linspace(transect_df["latitude"].min(), transect_df["latitude"].max(), 5),
    [90.0]
])
# ---- get number of bins
inpfc_labs = np.linspace(1, len(lat_bins) - 1, len(lat_bins) - 1).astype(int)
# ---- INPFC
# ---- Reduce transect data
sub_trans = transect_df[["latitude", "haul_num"]].loc[lambda x: x.haul_num > 0]
# ---- Bin
sub_trans["stratum_num"] = pd.cut(sub_trans["latitude"], 
                                  lat_bins, include_lowest=True, labels=inpfc_labs).astype(int)
# ---- Create dataframe
inpfc_strata = pd.DataFrame({
    "stratum_num": inpfc_labs[:-1],
    "northlimit_latitude": lat_bins[:len(inpfc_labs) - 1],
    "haul_start": sub_trans.groupby(["stratum_num"])["haul_num"].min().to_numpy(),
    "haul_end": sub_trans.groupby(["stratum_num"])["haul_num"].max().to_numpy()
})

# OPEN WRITER
writer = pd.ExcelWriter(Path(data_dir) / "geographic_strata.xlsx", engine="openpyxl")
# WRITE EXCEL : INPFC_STRATA SHEET
inpfc_strata.to_excel(
    writer,
    sheet_name="INPFC",
    index=False
)

# Repeat for the latitudinal extents for each KS group
transect_ks = transect_df.merge(ks_strata)

# ---- index strata
delta_strata = np.concatenate([np.array([0]), np.diff(transect_ks["stratum_num"])])
strata_idx = pd.Series(delta_strata).index
strata_idx_filt = strata_idx[delta_strata != 0].to_numpy()
# ---- Get order of strata via digitization
transect_ks["strata_grp"] = np.digitize(strata_idx, strata_idx[strata_idx_filt])
# ---- Reduce transect dataframe
transect_filt = (
    transect_ks.filter(["stratum_num", "strata_grp","haul_num", "latitude"])
    .drop_duplicates(["stratum_num", "strata_grp", "haul_num"])
)
# ---- Get min/max latitude for each group
# ------ strata grp max lat
max_lat = transect_filt.groupby(["strata_grp"])["latitude"].max()
first_haul = transect_filt.groupby(["strata_grp"])["haul_num"].min()
last_haul = transect_filt.groupby(["strata_grp"])["haul_num"].max()
geostrata_ks = transect_filt.copy().set_index("strata_grp")
geostrata_ks["northlimit_latitude"] = max_lat
geostrata_ks["haul_start"] = first_haul
geostrata_ks["haul_end"] = last_haul
# ---- reset index and reduce
geostrata_output = (
    geostrata_ks.reset_index()
    .filter(["stratum_num", "northlimit_latitude", "haul_start", "haul_end"])
    .drop_duplicates()
)

# WRITE EXCEL : GEO_STRATA SHEET
geostrata_output.to_excel(
    excel_writer=writer,
    sheet_name="stratification1",
    index=False
)

# CLOSE WRITER
writer.close()
####################################################################################################
# CREATE MESH GRID
# latitude
lat_pts = np.linspace(transect_df.latitude.min()*0.99, transect_df.latitude.max()*1.01, 100)
# longitude
lon_pts = np.linspace(transect_df.longitude.min()*0.99, transect_df.longitude.max()*1.01, 100)
# create grid
mesh_x, mesh_y = np.meshgrid(lat_pts, lon_pts)
# combine
mesh_comb = np.column_stack((mesh_x.flatten(), mesh_y.flatten()))
# convert to df
mesh_df = pd.DataFrame(mesh_comb, columns=["centroid_latitude", "centroid_longitude"])
# ---- add fraction
mesh_df["fraction_cell_in_polygon"] = 1.00

# WRITE EXCEL : GEO_STRATA SHEET
mesh_df.to_excel(
    excel_writer=Path(data_dir) / "kriging_mesh.xlsx",
    sheet_name="mesh_grid",
    index=False
)

####################################################################################################
# Isobath transform
# ---- y-axis resolution
isobath_y_pts = np.linspace(transect_df.latitude.min()*0.99, transect_df.latitude.max()*1.01, 25)
# ---- x-axis points
isobath_x_pts = np.linspace(transect_df.longitude.min()*0.99, transect_df.longitude.max()*1.01, 25)
# convert to df
isobath_df = pd.DataFrame({
    "longitude": isobath_x_pts,
    "latitude": isobath_y_pts,
})

# WRITE EXCEL : ISOBATH SHEET
isobath_df.to_excel(
    excel_writer=Path(data_dir) / "isobath.xlsx",
    sheet_name="isobath_smoothing",
    index=False
)

####################################################################################################
# PATH TO EXAMPLE DATA
import os 
import pandas as pd
import numpy as np
import math
import os 
from pathlib import Path
import matplotlib.pyplot as plt

# Get the current working directory where the notebook is running
notebook_dir = os.getcwd()

# Navigate to the example_data folder
data_dir = os.path.join(notebook_dir, "docs/example_notebooks/data")

# Path to the specific file
# ---- `init_config_path`
init_config_path = os.path.join(data_dir, "initialization_config.yml")
# ---- `survey_year_config_path`
survey_year_config_path = os.path.join(data_dir, "survey_year_config.yml")

# INITIALIZE ENVIRONMENT
from echopop.survey import Survey
# ---- Initialize Survey object
survey = Survey(init_config_path = init_config_path, 
                survey_year_config_path = survey_year_config_path)
# ---- Edit the root directory key
survey.config["data_root_dir"] = survey.config["data_root_dir"].replace("${PKG_PATH}", data_dir)

# Load acoustic data
survey.load_acoustic_data()
survey.load_survey_data()
survey.transect_analysis(species_id = 999999)

#########
import copy
from pathlib import Path
from typing import List, Literal, Optional, Union
stratum: Literal["inpfc", "ks"] = "ks"
verbose: bool = True
species_id: Union[float, list[float]] = 999999
exclude_age1: bool = True
self = survey

# Update settings to reflect the stratum definition
self.analysis["settings"].update(
    {
        "transect": {
            "age_group_columns": {
                "haul_id": "haul_no_age1" if exclude_age1 else "haul_all_ages",
                "nasc_id": "NASC_no_age1" if exclude_age1 else "NASC_all_ages",
                "stratum_id": "stratum_no_age1" if exclude_age1 else "stratum_all_ages",
            },
            "species_id": species_id,
            "stratum": stratum.lower(),
            "stratum_name": "stratum_num" if stratum == "ks" else "inpfc",
            "exclude_age1": exclude_age1,
        }
    }
)

input_dict = self.input
analysis_dict = self.analysis["transect"]
settings_dict = self.analysis["settings"]
configuration_dict = self.config
from echopop.utils import load as el, load_nasc as eln, message as em
from echopop.acoustics import aggregate_sigma_bs, nasc_to_biomass
from echopop.biology import (
    distribute_length_age,
    filter_species,
    fit_length_weight_relationship,
    fit_length_weights,
    impute_kriged_values,
    number_proportions,
    partition_transect_age,
    quantize_number_counts,
    quantize_weights,
    reallocate_kriged_age1,
    weight_proportions,
)
from echopop.spatial.transect import (
    edit_transect_columns,
    save_transect_coordinates,
    summarize_transect_strata,
    transect_spatial_features,
    correct_transect_intervals
)
from echopop.spatial.variogram import (
    empirical_variogram,
    initialize_initial_optimization_values,
    initialize_optimization_config,
    initialize_variogram_parameters,
    optimize_variogram,
)


# Filter out non-target species
length_data, specimen_data, catch_data = filter_species(
    [
        input_dict["biology"]["length_df"],
        input_dict["biology"]["specimen_df"],
        input_dict["biology"]["catch_df"],
    ],
    settings_dict["transect"]["species_id"],
)
# ---- For cases where all samples were aged (i.e. in `specimen_data` and absent from
# ---- `length_data`), these hauls are removed from `catch_data`
catch_data = catch_data[catch_data.haul_num.isin(length_data.haul_num)]

# Save the transect coordinate information
analysis_dict.update(
    {
        "coordinates": save_transect_coordinates(
            input_dict["acoustics"]["nasc_df"], settings_dict["transect"]
        )
    }
)

# Calculate mean sigma_bs per individual haul, KS stratum, and INPFC stratum
analysis_dict["acoustics"]["sigma_bs"].update(
    aggregate_sigma_bs(
        length_data, specimen_data, input_dict["spatial"], configuration_dict, settings_dict
    )
)
# analysis_dict["acoustics"]["sigma_bs"]
# Fit length-weight regression required for biomass calculation
analysis_dict["biology"]["weight"].update(
    fit_length_weight_relationship(
        specimen_data, input_dict["biology"]["distributions"]["length_bins_df"]
    )
)
# analysis_dict["biology"]["weight"]
# Count the number of specimens across age and length bins
analysis_dict["biology"]["distributions"].update(
    quantize_number_counts(
        specimen_data, length_data, stratum=settings_dict["transect"]["stratum"]
    )
)
# a = analysis_dict["biology"]["distributions"]["binned_aged_counts_df"].copy()
# b = analysis_dict["biology"]["distributions"]["binned_aged_counts_filtered_df"].copy()
# c = analysis_dict["biology"]["distributions"]["binned_unaged_counts_df"].copy()

# Calculate the number proportions
analysis_dict["biology"]["proportions"].update(
    {"number": number_proportions(analysis_dict["biology"]["distributions"])}
)
# analysis_dict["biology"]["proportions"]["number"]["unaged_length_proportions_df"]
# Sum the weights of both aged and unaged specimens across length and weight bins
# ---- Extract the length-weight fit
length_weight_df = analysis_dict["biology"]["weight"]["length_weight_regression"][
    "weight_fitted_df"
]
# ---- Quantize the weights
analysis_dict["biology"]["distributions"].update(
    {"weight": quantize_weights(specimen_data, length_data, length_weight_df)}
)

# Calculate the average weights among male, female, and all fish across strata
analysis_dict["biology"]["weight"].update(
    {
        "weight_stratum_df": fit_length_weights(
            analysis_dict["biology"]["proportions"]["number"],
            analysis_dict["biology"]["weight"],
        )
    }
)

# Calculate the weight proportions
analysis_dict["biology"]["proportions"].update(
    {
        "weight": weight_proportions(
            specimen_data,
            catch_data,
            analysis_dict["biology"]["proportions"]["number"],
            length_weight_df,
            analysis_dict["biology"]["distributions"]["weight"],
        )
    }
)

from echopop.analysis import (
    acoustics_to_biology,
    apportion_kriged_values,
    krige,
    process_transect_data,
    stratified_summary,
    variogram_analysis,
)

input_dict = self.input
analysis_dict = self.analysis["transect"]
configuration_dict = self.config
settings_dict = self.analysis["settings"]

# Convert NASC into number density (animals/nmi^2), biomass density (kg/nmi^2), abundance
# (# animals), and biomass (kg) for all fish, sexed (male/female) fish, and unsexed fish
# strata_adult_proportions, nasc_to_biology = nasc_to_biomass(
#     input_dict, analysis_dict, configuration_dict, settings_dict
# )
# Extract the necessary correct strata mean sigma_bs
sigma_bs_strata = analysis_dict["acoustics"]["sigma_bs"]["strata_mean_df"]

# Pull out the length-weight conversion for each stratum
length_weight_strata = analysis_dict["biology"]["weight"]["weight_stratum_df"]

# Get the name of the stratum column
stratum_col = settings_dict["transect"]["stratum_name"]

# Get group-specific columns
age_group_cols = settings_dict["transect"]["age_group_columns"]

# Get group-specific column names and create conversion key
name_conversion_key = {age_group_cols["haul_id"]: "haul_num", age_group_cols["nasc_id"]: "nasc"}
# ---- Update if the stratum is not equal to INPFC
if settings_dict["transect"]["stratum"] != "inpfc":
    name_conversion_key.update({age_group_cols["stratum_id"]: stratum_col})

# Rename columns
# ---- Extract NASC data
nasc_data = input_dict["acoustics"]["nasc_df"].copy()
# ---- Change names
nasc_data.rename(columns=name_conversion_key, inplace=True)

# Correct the acoustic survey transect intervals
nasc_interval_df = correct_transect_intervals(nasc_data)

from echopop.biology import age1_metric_proportions

# Select the appropriate NASC column based on the inclusion or exclusion of age-1 fish
if settings_dict["transect"]["exclude_age1"]:
    # ---- Calculate age-1 NASC and weight proportions
    age1_proportions = age1_metric_proportions(
        input_dict["biology"]["distributions"],
        analysis_dict["biology"]["proportions"],
        configuration_dict["TS_length_regression_parameters"]["pacific_hake"],
        settings_dict,
    )
    # ---- Calculate adult proportions
    # -------- Initialize dataframe
    adult_proportions = age1_proportions.copy()
    # -------- Invert proportions
    # ------------ Number
    adult_proportions["number_proportion"] = 1 - age1_proportions["number_proportion"]
    # ------------ Weight
    adult_proportions["weight_proportion"] = 1 - age1_proportions["weight_proportion"]
    # ------------ Weight
    adult_proportions["nasc_proportion"] = 1 - age1_proportions["nasc_proportion"]

else:
    # ---- Assign filled adult proportions dataframe
    adult_proportions = pd.DataFrame(
        {
            f"{stratum_col}": np.unique(nasc_interval_df[stratum_col]),
            "number_proportion": np.ones(len(np.unique(nasc_interval_df[stratum_col]))),
            "weight_proportion": np.ones(len(np.unique(nasc_interval_df[stratum_col]))),
            "nasc_proportion": np.ones(len(np.unique(nasc_interval_df[stratum_col]))),
        }
    )


# Distribute abundance and biomass estimates over length and age bins
analysis_dict["biology"]["population"].update(
    {
        "tables": distribute_length_age(
            nasc_to_biology, analysis_dict["biology"]["proportions"], settings_dict
        )
    }
)
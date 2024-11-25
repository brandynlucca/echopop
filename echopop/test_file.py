from echopop.survey import Survey

survey = Survey(init_config_path = "C:/Users/Brandyn Lucca/Documents/GitHub/echopop/config_files/initialization_config.yml",
                survey_year_config_path = "C:/Users/Brandyn Lucca/Documents/GitHub/echopop/config_files/survey_year_2019_config.yml")
survey.load_acoustic_data()
survey.load_survey_data()
survey.transect_analysis()
survey.fit_variogram()
survey.kriging_analysis(best_fit_variogram=True)
survey.stratified_analysis()

dataset = survey.results["variogram"]
survey.analysis["settings"]["kriging"]

import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from IPython.display import display

from echopop.analysis import (
    acoustics_to_biology,
    apportion_kriged_values,
    krige,
    process_transect_data,
    stratified_summary,
    variogram_analysis,
)
from echopop.core import DATA_STRUCTURE
from echopop.graphics import plotting as egp, variogram_interactive as egv
from echopop.spatial.projection import transform_geometry
from echopop.spatial.transect import edit_transect_columns
from echopop.utils import load as el, load_nasc as eln, message as em
from echopop.utils.load import dataset_integrity
import numpy as np
import pandas as pd

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
from echopop.spatial.krige import kriging
from echopop.spatial.mesh import crop_mesh, mesh_to_transects, stratify_mesh
from echopop.spatial.projection import transform_geometry
from echopop.spatial.transect import (
    edit_transect_columns,
    save_transect_coordinates,
    summarize_transect_strata,
    transect_spatial_features,
)
from echopop.spatial.variogram import (
    empirical_variogram,
    initialize_initial_optimization_values,
    initialize_optimization_config,
    initialize_variogram_parameters,
    optimize_variogram,
)
from echopop.statistics import stratified_transect_statistic
from echopop.utils.validate_dict import (
    KrigingAnalysis,
    KrigingParameterInputs,
    MeshCrop,
    VariogramBase,
    VariogramEmpirical,
)

self = survey
variogram_parameters = {}
optimization_parameters = {}
model: Union[str, List[str]] = ["bessel", "exponential"]
n_lags: int = 30
azimuth_range: float = 360.0
standardize_coordinates: bool = True
force_lag_zero: bool = True
initialize_variogram: Union[List[str], Dict[str, Any]] = [
    "nugget",
    "sill",
    "correlation_range",
    "hole_effect_range",
    "decay_power",
]
variable: Literal["biomass"] = "biomass"
verbose: bool = True

transect_dict = self.analysis["transect"]
settings_dict = self.analysis["settings"]["variogram"]
isobath_df = self.input["statistics"]["kriging"]["isobath_200m_df"]


kriging_parameters: Dict[str, Any] = {}
cropping_parameters: Dict[str, Any] = {}
coordinate_transform: bool = True
extrapolate: bool = False
best_fit_variogram: bool = True
variable: Literal["biomass"] = "biomass"
variogram_parameters: Optional[Dict[str, Any]] = None
verbose: bool = True


# Check dataset integrity
dataset_integrity(self.input, analysis="kriging")
self.analysis["settings"]["kriging"]
self.analysis["settings"]["variogram"]
# Populate settings dictionary with input argument values/entries
self.analysis["settings"].update(
    {
        "kriging": {
            "best_fit_variogram": best_fit_variogram,
            "cropping_parameters": {**cropping_parameters},
            "extrapolate": extrapolate,
            "kriging_parameters": {**kriging_parameters},
            "standardize_coordinates": coordinate_transform,
            "variable": variable,
            "verbose": verbose,
        },
    },
)

# Inherited settings/configurations (contingent on previously executed methods)
self.analysis["settings"]["kriging"].update(
    {
        # ---- From `self.config`
        "projection": self.config["geospatial"]["init"],
        # ---- From `self.transect_analysis` settings
        "exclude_age1": self.analysis["settings"]["transect"]["exclude_age1"],
        "stratum": self.analysis["settings"]["transect"]["stratum"],
    },
)

self.analysis["settings"]["kriging"].update(
    {
        "stratum_name": (
            "stratum_num"
            if self.analysis["settings"]["kriging"]["stratum"] == "ks"
            else "inpfc"
        ),
        "variogram_parameters": (
            {
                **self.input["statistics"]["variogram"]["model_config"],
                **variogram_parameters,
            }
            if (
                variogram_parameters is not None
                and "model_config" in self.input["statistics"]["variogram"]
            )
            else (
                {
                    **self.input["statistics"]["variogram"]["model_config"],
                }
                if (
                    not best_fit_variogram
                    and not variogram_parameters
                    and "model_config" in self.input["statistics"]["variogram"]                    
                )
                else (
                    {
                        **self.input["statistics"]["variogram"]["model_config"],
                        **self.results["variogram"]["model_fit"],
                    }
                    if best_fit_variogram is True
                    else {}
                )
            )
        ),
    }
)
# ---- Further append variogram parameters if they were ran
if "variogram" in self.analysis["settings"]:
    self.analysis["settings"]["kriging"]["variogram_parameters"].update(
        **self.analysis["settings"]["variogram"]
    )

# Run kriging analysis
# ----> Generates a georeferenced dataframe for the entire mesh grid, summary statistics,
# ----> and adds intermediate data products to the analysis attribute
# ---- Run kriging algorithm
kriged_results, self.analysis = krige(
    self.input, self.analysis, self.analysis["settings"]["kriging"]
)
input_dict = self.input
analysis_dict = self.analysis
settings_dict = self.analysis["settings"]["kriging"]

import pandas as pd
df = pd.DataFrame(self.analysis["settings"]["kriging"]["kriging_parameters"], index=[0])
df_T = df.T
df_transformed = pd.DataFrame({
    'Column Names': df.columns,
    'Values': df.iloc[0].values
})

df_transformed.to_excel("C:/Users/Brandyn Lucca/Documents/test_file.xlsx", index=False, header=False)

from pydantic import BaseModel, Field, ConfigDict

class MyModel(BaseModel):
    correlation_range: float = Field(alias='vario.lscl')
    
    # Updated configuration
    model_config = ConfigDict(populate_by_name=True)


# Example usage
# Input data may use either column name
data_with_lscl = {'vario.lscl': 10.5}
data_with_correlation_range = {'correlation_range': 10.5}

# Both will parse correctly
model1 = MyModel(**data_with_lscl)
model2 = MyModel(**data_with_correlation_range)

from echopop.core import BIODATA_HAUL_MAP, DATA_STRUCTURE, LAYER_NAME_MAP, NAME_CONFIG
from echopop.utils.data_structure_utils import map_imported_datasets
from echopop.utils.validate_df import DATASET_DF_MODEL
from echopop.utils.validate_dict import CONFIG_DATA_MODEL, CONFIG_INIT_MODEL

input_dict = self.input
configuration_dict = self.config
dataset_type = ["kriging"]
dataset = "kriging"
datalayer = 'vario_krig_para'

sheets = sheet_name[0]


# Read Excel file into memory and then transpose
df_initial = pd.read_excel(file_name, header=None).T

# Take the values from the first row and redfine them as the column headers
df_initial.columns = df_initial.iloc[0]
df_initial = df_initial.drop(0)

class VarioKrigingPara(BaseDataFrame):
    """
    Haul-transect map DataFrame

    Parameters
    ----------
    hole: float
        Length scale or range of the hole effect.
    lscl: float
        The relative length scale, or range at which the correlation between points becomes
        approximately constant.
    nugt: float
        The y-intercept of the variogram representing the short-scale (i.e. smaller than the lag
        resolution) variance.
    powr: float
        The exponent used for variogram models with exponentiated spatial decay terms.
    res: float
        The (scaled) distance between lags.
    sill: float
        The total variance where the change autocorrelation reaches (or nears) 0.0.
    ratio: float
        The directional aspect ratio of anisotropy.
    srad: float
        The adaptive search radius used for kriging.
    kmin: float
        The minimum number of nearest kriging points.
    kmax: float
        The maximum number of nearest kriging points.
    """

    y_offset: Series[float] = Field(ge=-90.0, le=90.0, nullable=False, alias="dataprep.y_offset")
    corr: Series[float] = Field(ge=0.0, nullable=False, alias="vario.corr")
    hole: Series[float] = Field(ge=0.0, nullable=False, alias="vario.hole")
    lscl: Series[float] = Field(ge=0.0, nullable=False, alias="vario.lscl")
    nugt: Series[float] = Field(ge=0.0, nullable=False, alias="vario.nugt")
    powr: Series[float] = Field(ge=0.0, nullable=False, alias="vario.powr")
    range: Series[float] = Field(ge=0.0, nullable=False, alias="vario.range")
    res: Series[float] = Field(gt=0.0, nullable=False, alias="vario.res")
    sill: Series[float] = Field(ge=0.0, nullable=False, alias="vario.sill")
    ytox_ratio: Series[float] = Field(nullable=False, alias="vario.ytox_ratio")
    ztox_ratio: Series[float] = Field(nullable=False, alias="vario.ztox_ratio")
    blk_nx: Series[int] = Field(gt=0, nullable=False, alias="krig.blk_nx")
    blk_ny: Series[int] = Field(gt=0, nullable=False, alias="krig.blk_ny")
    blk_nz: Series[int] = Field(gt=0, nullable=False, alias="krig.blk_nz")
    dx0: Series[float] = Field(ge=-180.0, le=180.0, nullable=False, alias="krig.dx0")
    dx: Series[float] = Field(nullable=False, alias="krig.dx")
    dy0: Series[float] = Field(ge=-90.0, le=90.0, nullable=False, alias="krig.dy0")
    dy: Series[float] = Field(nullable=False, alias="krig.dy")
    dz: Series[float] = Field(nullable=False, alias="krig.dz")
    elim: Series[float] = Field(nullable=False, alias="krig.elim")
    eps: Series[float] = Field(nullable=False, alias="krig.eps")
    kmax: Series[int] = Field(gt=0, nullable=False, alias="krig.kmax")
    kmin: Series[int] = Field(gt=0, nullable=False, alias="krig.kmin")
    nx: Series[int] = Field(gt=0, nullable=False, alias="krig.nx")
    ny: Series[int] = Field(gt=0, nullable=False, alias="krig.ny")
    nz: Series[int] = Field(gt=0, nullable=False, alias="krig.nz")
    ratio: Series[float] = Field(nullable=False, alias="krig.ratio")
    srad: Series[float] = Field(gt=0.0, nullable=False, alias="krig.srad")
    x_res: Series[float] = Field(nullable=False, alias="krig.x_res")
    xmin: Series[float] = Field(nullable=False, alias="krig.xmin")
    xmax: Series[float] = Field(nullable=False, alias="krig.xmax")
    xmin0: Series[float] = Field(ge=-180.0, le=180.0, nullable=False, alias="krig.xmin0")
    xmax0: Series[float] = Field(ge=-180.0, le=180.0, nullable=False, alias="krig.xmax0")
    y_res: Series[float] = Field(nullable=False, alias="krig.y_res")
    ymin: Series[float] = Field(nullable=False, alias="krig.ymin")
    ymax: Series[float] = Field(nullable=False, alias="krig.ymax")
    ymin0: Series[float] = Field(ge=-90.0, le=90.0, nullable=False, alias="krig.ymin0")
    ymax0: Series[float] = Field(ge=-90.0, le=90.0, nullable=False, alias="krig.ymax0")

import pandas as pd
import pandera as pa
from pandera.engines.pandas_engine import PydanticModel
from pydantic import BaseModel, Field

class VarioKrigingPara(BaseDataFrame):
    correlation_range: Series[float]
        
    @pa.check_input
    def transform_columns(data: pd.DataFrame) -> pd.DataFrame:
        data.rename(columns={
            "krig.kmax": "kmax",
            "krig.kmin": "kmin",
            "krig.ratio": "anisotropy",
            "krig.srad": "search_radius",
            "vario.hole": "hole_effect_range",
            "vario.lscl": "correlation_range",
            "vario.nugt": "nugget",
            "vario.powr": "decay_power",
            "vario.res": "lag_resolution",            
            }, inplace=True)
        return data
    
    
VarioKrigingPara.transform_columns(data_with_lscl)
print(VarioKrigingPara(data_with_lscl))
schema = pa.DataFrameSchema({
    "str_column": pa.Column(str),
    "float_column": pa.Column(float),
    "int_column": pa.Column(int),
    "date_column": pa.Column(pa.DateTime),
})

schema = pa.DataFrameSchema({
    # `Echopop` names
    "correlation_range": pa.Column(float, default=None, required=False),
    # Deprecated names
    "krig.kmax": pa.Column(float, default=None, required=False),
    "krig.kmin": pa.Column(float, default=None, required=False),
    "krig.ratio": pa.Column(float, default=None, required=False),
    "krig.srad": pa.Column(float, default=None, required=False),
    "vario.hole": pa.Column(float, default=None, required=False),
    "vario.lscl": pa.Column(float, default=None, required=False),
    "vario.nugt": pa.Column(float, default=None, required=False),
    "vario.powr": pa.Column(float, default=None, required=False),
    "vario.res": pa.Column(float, default=None, required=False),  
    "vario.lscl": pa.Column(float, default=None, required=False)
})

@pa.check_input(schema)
def transform_columns(data: pd.DataFrame) -> pd.DataFrame:
    data.rename(columns={
        "krig.kmax": "kmax",
        "krig.kmin": "kmin",
        "krig.ratio": "anisotropy",
        "krig.srad": "search_radius",
        "vario.hole": "hole_effect_range",
        "vario.lscl": "correlation_range",
        "vario.nugt": "nugget",
        "vario.powr": "decay_power",
        "vario.res": "lag_resolution",            
        }, inplace=True)
    return data

data_with_lscl = pd.DataFrame({"vario.lscl": [10.5]})
data_with_correlation_range = pd.DataFrame({"correlation_range": [10.5]})

class MyModel(BaseDataFrame):
    anisotropy: Series[float] = pa.Field(ge=0.0, nullable=False)
    correlation_range: Series[float] = pa.Field(ge=0.0, nullable=False)
    decay_power: Series[float] = pa.Field(ge=0.0, nullable=False)
    hole_effect_range: Series[float] = pa.Field(ge=0.0, nullable=False)
    kmax: Series[int] = pa.Field(gt=0, nullable=False)
    kmin: Series[int] = pa.Field(gt=0, nullable=False)
    lag_resolution: Series[float] = pa.Field(gt=0.0, nullable=False)
    model: Series = pa.Field(nullable=False, metadata=dict(types=[str, List[str]]))
    n_lags: Optional[Series[int]] = pa.Field(gt=0, nullable=False)
    search_radius: Series[float] = pa.Field(gt=0.0, nullable=False)
    sill: Series[float] = pa.Field(ge=0.0, nullable=False)
    
    @check(
        "model",
        name="element-wise datatypes",
        error="Model must belong to ",
    )
    def validate_model(cls, model: Union[str, List[str]]) -> bool:
        if isinstance(model, list):
            # Test lists
            composite_models = [["exponential", "bessel"]]
                       
            return Series([value in [["exponential", "bessel"]] for value in model])
            
            return Series[any([
                c == v.values[0] for c in composite_models
            ])]

    @classmethod
    def transform_columns(cls, data: pd.DataFrame) -> pd.DataFrame:
        data.rename(columns={
            "krig.kmax": "kmax",
            "krig.kmin": "kmin",
            "krig.ratio": "anisotropy",
            "krig.srad": "search_radius",
            "vario.hole": "hole_effect_range",
            "vario.lscl": "correlation_range",
            "vario.nugt": "nugget",
            "vario.powr": "decay_power",
            "vario.res": "lag_resolution",
            "vario.sill": "sill",            
            }, inplace=True)
        return data    



a = df_initial.copy()
b = MyModel.transform_columns(df_initial)
if "model" in b.columns:
    b.at[1, "model"] = ["exponential", "bessel"]
else:
    b["model"] = ["exponential", "bessel"]
c = MyModel.validate_df(b)

c = composite_models[0]
c
v = b["model"]
type(v)
type(c)
v = b["model"]
a.model[1] == ["exponential", "bessel"]
type(a.model[1])


df = pd.DataFrame(data={'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
df.at[1, 'B'] = ['m', 'n']

a["model"].dtype


MyModel.validate(data_with_lscl).rename(columns=rename_fields)

MyModel.validate_df(data_with_correlation_range)
MyModel.validate_df(data_with_lscl)

transform_columns(data_with_lscl)
transform_columns(data_with_correlation_range)

print(VarioKrigingPara.validate(data_with_correlation_range))
PydanticSchema.validate(pd.DataFrame({"vario.lscl": [-1.0]}))
print(validated_df1)
print(validated_df2)   
    
MySchema.__annotations__
# Example data
data_with_lscl = pd.DataFrame({"vario.lscl": [10.5]})
data_with_correlation_range = pd.DataFrame({"correlation_range": [10.5]})

# Validate both variations of input
validated_df1 = MySchema.validate(data_with_lscl)
validated_df2 = MySchema.validate(data_with_correlation_range)

print(validated_df1)
print(validated_df2)

class MyDataFrameModel(DataFrameModel):
    correlation_range: Series[float] = Field(alias="vario.lscl")

# Example data


# Validate both variations
validated_df1 = MyDataFrameModel(data_with_lscl)
validated_df2 = MyDataFrameModel(data_with_correlation_range)

print(validated_df1)
print(validated_df2)
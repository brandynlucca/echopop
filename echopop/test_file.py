from echopop.survey import Survey

survey = Survey(init_config_path = "C:/Users/Brandyn/Documents/GitHub/echopop/config_files/initialization_config.yml",
                survey_year_config_path = "C:/Users/Brandyn/Documents/GitHub/echopop/config_files/survey_year_2019_config.yml")
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


settings_dict["kriging_parameters"]

# pd.DataFrame({"model": [["exponential", "bessel"]]}).to_excel("C:/Users/Brandyn/Documents/text.xlsx", index=False)
# pd.read_excel("C:/Users/Brandyn/Documents/text.xlsx")

from pandera import check, Field, DataFrameModel
from pandera.typing import Series
from echopop.spatial.variogram import VARIOGRAM_MODELS

# Define your allowed model types
ALLOWED_MODELS = {"exponential", "gaussian", "bessel"}

def is_valid_model(model: Union[str, List[str]]) -> bool:
    if isinstance(model, str):
        return model in VARIOGRAM_MODELS["single"]
    elif isinstance(model, list):
        # ---- Convert from tuples to lists
        comp_mods = [list(m) for m in VARIOGRAM_MODELS["composite"]]
        return sorted(model) in comp_mods
    elif isinstance(model, (float, int)):
        return True
    return False

class VarioKrigingPara(BaseDataFrame):
    """
    DataFrame model for validating haul-transect map parameters.
    """
    model: Optional[Series] = Field(default=None, metadata=dict(types=[List[str], str]))

    @check("model", name="Check model")
    def check_model(cls, s: Union[str, List[str]]) -> bool:     
        return s.map(is_valid_model)

comp_mods

sorted(["bessel", "exponential"])
sorted(["exponential", "bessel"]) in comp_mods


data = pd.DataFrame({"model": ["exponential", "bessel"]})
s = data.model
    
VarioKrigingPara.validate_df(pd.DataFrame({"model": [["bessel", "exponential"], "exponential"]}))
VarioKrigingPara.validate_df(pd.DataFrame({"model": [["exponential", "bessel"]]}))
df = pd.DataFrame({
    "model": [["bessel", "exponential"]]
})

cls = VarioKrigingPara
data = df

# dd = pd.DataFrame({"model": [["exponential", "bessel"]]})
data = pd.DataFrame({"model": [1]})
VarioKrigingPara.validate_df(dd)
# ---- Create copy
df = data.copy()
# ---- Get the original annotations
default_annotations = copy.deepcopy(cls.__annotations__)
# ---- Format the column names
df.columns = [col.lower() for col in df.columns]
# ---- Get column types
column_types = cls.get_column_types()
# ---- Initialize invalid index
invalid_idx = {}
# ---- Initialize column names
valid_cols = []
# ---- Find all indices where there are violating NaN/-Inf/Inf values
for column_name, dtype in column_types.items():
    # ---- Apply coercion based on column patterns
    for col in df.columns:
        # ---- Regular expression matching
        if re.match(column_name, col):
            # ---- Retrieve the column name
            col_name = re.match(column_name, col).group(0)
            # ---- Collect, if valid
            if cls.to_schema().columns[column_name].regex:
                valid_cols.append(col)
            else:
                valid_cols.append(col_name)
            # ---- Check if null values are allowed
            if not cls.to_schema().columns[column_name].nullable:
                # ---- Get the violating indices
                invalid_idx.update(
                    {
                        col: df.index[
                            df[col].isna() | df[col].isin([np.inf, -np.inf])
                        ].to_list()
                    }
                )
# ---- Initialize the list
invalid_lst = []
# ---- Extend the values
invalid_lst.extend(
    [
        value
        for key in invalid_idx.keys()
        if invalid_idx[key] is not None and key in valid_cols
        for value in invalid_idx[key]
    ]
)
# ---- If the indices are invalid, but can be dropped then drop them
df.drop(invalid_lst, axis=0, inplace=True)
# ---- Reset index
if not df.empty:
    df.reset_index(inplace=True, drop=True)
col_types = column_types
column_name = "northlimit_latitude"
dtype = col_types[column_name]
typing = dtype[0]

correct_type = None
df["test"] = ["a"]
col = "model"
dtype = [List[int], int]
df = pd.DataFrame(dict(model=[["exponential", "bessel"]]))
dtype = [List[str], str]

errors_coerce = pd.DataFrame(dict(Column=[], error=[]))


for typing in dtype:
    # ---- Attempt coercion, if allowed
    
    try:
        df[col] = cls._DTYPE_COERCION.get(typing)(df[col])
    except Exception:
        continue
    # ---- Run test
    test = cls._DTYPE_TESTS.get(typing)(df[col])

if not test:
    message = ValueError(
        f"{col.capitalize()} column must be a Series of '{str(dtype)}' "
        f"values. Series values could not be automatically coerced."      
    )
    errors_coerce = pd.concat(
        [errors_coerce, pd.DataFrame(dict(Column=col, error=message))]
    )

class_schema[col].coerce

KrigedMesh.to_schema().columns[".*fraction.*"].coerce

data = pd.DataFrame(dict(haul=[1], northlimit_latitude=[1.0], stratum=[1]))
cls = GeoStrata
col_types = column_types
column_name = "haul"
dtype = col_types[column_name]
col = "haul"
typing = dtype[0]
df["haul"] = pd.Series([np.sum])
df["stratum"] = pd.Series([dict()])
dtype = [int, int]


dir(cls)
cls.__dir__(cls)

_DTYPE_COERCION = {
    int: lambda v: v.astype(np.float64, errors="ignore").astype(np.int64, errors="ignore"),
    float: lambda v: v.astype(np.float64, errors="ignore"),
    # str: lambda v: v.astype(str, errors="ignore"),
    str: lambda v: v.astype(str),
    List[str]: lambda v: [[str(i) for i in x] for x in v]
}
_DTYPE_TESTS = {
    int: (
        lambda v: pd.api.types.is_integer_dtype(v)
        or (pd.api.types.is_numeric_dtype(v) and (v % 1 == 0).all())
    ),
    float: lambda v: pd.api.types.is_float_dtype(v),
    str: lambda v: pd.api.types.is_string_dtype(v, dtype=str) and not isinstance(v, list),
    List[str]: lambda v: all(
        isinstance(x, list) and all(isinstance(xi, str) for xi in x)
        for x in v
    )
}

# Initialize a DataFrame
errors_coerce = pd.DataFrame(dict(Column=[], error=[]))

# Get the class schema
class_schema = cls.to_schema().columns

_DTYPE_TESTS = {
    int: (
        lambda v: pd.api.types.is_integer_dtype(v)
        or (pd.api.types.is_numeric_dtype(v) and (v % 1 == 0).all())
    ),
    float: lambda v: pd.api.types.is_float_dtype(v),
    str: lambda v: pd.api.types.is_string_dtype(v) and not isinstance(v, list),
    List[str]: lambda v: all(
        isinstance(x, list) and all(isinstance(xi, str) for xi in x)
        for x in v
    )
}

_DTYPE_TESTS[List[str]](pd.Series([1.0, 2.0, "a"]))


_DTYPE_COERCION = {
    int: lambda v: v.astype(np.float64, errors="ignore").astype(np.int64, errors="ignore"),
    float: lambda v: v.astype(np.float64, errors="ignore"),
    str: lambda v: v.astype(str, errors="ignore"),
    List[str]: lambda v: [[str(i) for i in x] if isinstance(v, list) else x for x in v]
}

cls._DTYPE_COERCION = _DTYPE_COERCION

_DTYPE_COERCION[List[str]](pd.Series([1.0, 2.0, "a"]))
cls._DTYPE_COERCION[List[str]](pd.Series([1.0, 2.0, "a"]))
col_types = column_types

# cls = GeoStrata
cls = VarioKrigingPara
df = pd.DataFrame({"model": [["exponential", "bessel"]]})

cls = GeoStrata

col_types = column_types
column_name = '.*fraction.*'
dtype = col_types[column_name]
col = "fraction"


input = pd.DataFrame(
                dict(haul=[1.0, 2.0, 3.0], fraction=[0.0, 0.0, 1.0], stratum=[1, "2a", "3b"])
            )
output = pd.DataFrame(dict(latitude=[-1.0, 0.0, 1.0], longitude=[-1.0, 0.0, 1.0]))

KSStrata.validate_df(input)


cls = KSStrata
data = input

IsobathData.validate_df(input) == output

data = input
GeoStrata.validate_df(df)
column_name = "stratum"
dtype = int
col = "stratum"

# Initialize a DataFrame
errors_coerce = pd.DataFrame(dict(Column=[], error=[]))

# Get the class schema
class_schema = cls.to_schema().columns

# Get the dtypes for each column of the input data
column_dtypes = {col: pd.api.types.infer_dtype(df[col]) for col in df.columns}

pd.api.types.is_integer_dtype(df[col])
(pd.api.types.is_numeric_dtype(df[col]) and (df[col] % 1 == 0).all())




# Coerce the data and test the data
for column_name, dtype in col_types.items():
    # ---- Iterate across the data columns
    for col in df.columns:
        # ---- Apply coercion and tests based on column patterns
        if re.match(column_name, col):
            # ---- Initialize error flag
            error_flag = False
            # ---- Alternate method that equates to `typing.Union` that is otherwise disallowed 
            if isinstance(dtype, list):
                # ---- Run an initial test with coercion, if configured
                tests = [d for d in dtype
                         if (cls._DTYPE_TESTS[d](cls._DTYPE_COERCION[d](df[col])) 
                             and class_schema[col].coerce)
                         or (cls._DTYPE_TESTS[d](df[col]) and not class_schema[col].coerce)]
                # ---- If only a single test succeeds (no coercion necessary)
                if len(tests) == 1:
                    # ---- Coerce
                    if class_schema[col].coerce:
                        df[col] = cls._DTYPE_COERCION[tests[0]](df[col])
                    # ---- Update class annotation
                    cls.__annotations__[col] = Series[tests[0]]
                    continue
                # ---- If multiple tests succeed
                elif len(tests) > 1:
                    # ---- Retest using the uncoerced data
                    retests = [d for d in tests if cls._DTYPE_TESTS[d](df[col])]
                    if len(retests) > 1:
                        # ---- Coerce
                        if class_schema[col].coerce:
                            df[col] = cls._DTYPE_COERCION[retests[0]](df[col])
                        # ---- Update class annotation giving priority to the leading dtype
                        cls.__annotations__[col] = Series[retests[0]]
                    else:
                        # ---- Assign error
                        error_flag = True
            # ----  When not `typing.Union`
            else:
                # ---- Coerce
                if class_schema[col].coerce:
                    df[col] = cls._DTYPE_COERCION[dtype](df[col])
                # ---- Test
                if cls._DTYPE_TESTS.get(dtype)(df[col]):
                    continue
                else:
                    # ---- Assign error
                    error_flag = True
            # ---- If the error flag was raised, raise the error message
            if error_flag:
                # ---- Join dtypes
                message = TypeError(
                    f"{col.capitalize()} column must be a Series of '{str(dtype)}' "
                    f"values. Series values could not be automatically coerced."      
                )
                # ---- Concatenate
                errors_coerce = pd.concat(
                    [errors_coerce, pd.DataFrame(dict(Column=[col], error=[message]))]
                )
                
                # ---- Distribute data for each dtype
                dtype_dict = {d: df.loc[:10, col] for d in dtype}
                # ---- Attempt coercion, if enabled/allowed
                if class_schema[col].coerce:
                    # ---- Create coercion dictionary (this generates valid coercion options)
                    dtype_dict.update({
                        d: cls._DTYPE_COERCION[d](dtype_dict[d]) for d in dtype
                    })                    
                # ---- Winnow the applicable dtype keys
                tests = [d for d in dtype if cls._DTYPE_TESTS[d](dtype_dict[d])]
                # ---- If only a single test succeeds
                if len(tests) == 1:
                    # ---- Update class annotation
                    cls.__annotations__[col] = Series[tests[0]]
                    break
                # ---- If multiple still remain
                elif len(tests) > 1:
                    # ---- Let `pandas` attempt inferences
                    
                    
                # ---- Attempt coercion, if enabled/allowed
                if class_schema[col].coerce:
                    # ---- Create coercion dictionary
                    coerce_dict = {d: cls._DTYPE_COERCION[d](df[col]) for d in dtype}
                    # ---- Attempt second test
                    tests = [d for d in dtype if cls._DTYPE_TESTS[d](coerce_dict[d])]
                    
                    
                # ---- Initialize the test result to False
                test = False
            
    

list_values = [c for c, v in col_types.items() if isinstance(v, list)]

errors_coerce
df.dtypes
type(df["stratum"][0])
_DTYPE_COERCION[int](pd.Series(["a"]))
jed = lambda v: pd.api.types.is_object_dtype(v, dtype=str)
jed(df.stratum)

df.stratum
pd.api.types.is_string_dtype(np.array(["a", "b"], dtype=str))

print(f"YEA {typing}")

input = pd.DataFrame(
                dict(haul=[1, 2, 3], northlimit_latitude=[-91.0, 0.0, 1.0], stratum=[1, 2, 3])
            )
cls = GeoStrata
exception = "greater_than_or_equal_to(-90.0)"

GeoStrata.judge(input)

with pytest.raises(SchemaError, match=re.escape(exception)):
    assert GeoStrata.validate_df(input)
    
GeoStrata.validate_df(data) == pd.DataFrame(
                dict(haul=[1, 2, 3], northlimit_latitude=[-1.0, 0.0, 1.0], stratum=[1, 2, 3])
            )

df["test"] = ["a"]
col = "model"
dtype = [List[int], int]
df = pd.DataFrame(dict(model=[["exponential", "bessel"]]))
dtype = [List[str], str]


data = pd.DataFrame(dict(haul=[1, 2, 3], northlimit_latitude=[-1, 0, 1], stratum=[1, 2, 3]))

jam = lambda v: pd.api.types.is_object_dtype(v) and not isinstance(v, list)
jam(df.model)

jam = lambda v: all(
    isinstance(x, list) and all(isinstance(i, str) for i in x) if isinstance(x, list) else isinstance(x, str)
    for x in v
)
jam(df.model)

dat = pd.DataFrame({"model": [["exp", "bes"], ["cos", 2], ["ah", 1]]})
jak = lambda v: [[str(i) for i in x] for x in v]
jak(dat.model)

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
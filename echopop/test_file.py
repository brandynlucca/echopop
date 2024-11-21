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
import inspect
from typing import Any, Dict, Literal, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import verde as vd
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.ticker import FixedLocator
import re
from typing import Any, Dict, Literal, Optional

import cartopy.feature as cfeature
from cartopy.crs import PlateCarree, Projection
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, SerializeAsAny
from echopop.graphics.validate_plot import GeoConfigPlot, SpatialPlot
from echopop.graphics.plotting import (
    scale_sizes, get_survey_bounds, add_alongtransect_data, add_colorbar,
    add_heatmap_grid, add_transect_lines, plot_age_length_distribution, plot_mesh, plot_transect,
    PLOT_MAP, validate_plot_args, apply_aspect_ratio, format_axes, format_heatmap_ticks, 
    spatial_mesh, validate_plot_args, prune_args
)


from echopop import Survey

init_config = "C:/Users/Brandyn Lucca/Documents/GitHub/echopop/config_files/initialization_config.yml"
file_config = "C:/Users/Brandyn Lucca/Documents/GitHub/echopop/config_files/survey_year_2019_config.yml"
survey = Survey(init_config, file_config)
survey.load_survey_data(verbose=False)
survey.load_acoustic_data(verbose=False)
survey.transect_analysis(verbose=False)
survey.fit_variogram(verbose=False)
survey.kriging_analysis(variogram_parameters={"n_lags": 30}, 
                        variable="biomass_density", verbose=False)

from cartopy.crs import PlateCarree
from cartopy.feature import NaturalEarthFeature

# Define the plot projection
plot_projection = PlateCarree()

# Define the coastline shape object (whatever floats your boat doesn't sink mine...)
eyesore_coastline = NaturalEarthFeature(
    category="physical",
    name="coastline",
    scale="110m",
    linewidth=1.0,
    facecolor="limegreen",
    edgecolor="cyan",
)

self = survey
kind = "mesh"
variable = "biomass"
plot_parameters = {"geo_config": {"coastline": eyesore_coastline}}
plot_type = "hexbin"


# Get associated plotting function information
plot_info = egp.PLOT_MAP(self, kind)

# Initialize 'parameters' dictionary
parameters = plot_parameters.copy()

# Proceed with plotting
# ---- Type: spatial
if plot_info["type"] == "spatial":
    # ---- Get the geospatial configuration
    geo_config = self.config["geospatial"].copy()
    # ---- Create copy of user-defined geospatial configuration
    geo_param = parameters["geo_config"].copy()
    # ---- Update the parameterization
    parameters["geo_config"].update({**geo_config, **geo_param})
    
parameters["geo_config"]

#     # ---- Get the coastline, if it exists, from `plot_parameters`
#     if parameters.get("geo_config"):
#         geo_config.update(parameters.get("geo_config", None))
#     # ---- Update the parameterization
#     parameters.update(dict(geo_config=geo_config))

# Add the primary arguments into the dictionary
parameters.update(dict(kind=kind, plot_type=plot_type, variable=variable))
GeoConfigPlot.create()
SpatialPlot.create(**parameters)

cls = HexbinPlot
values = {"kind": "mesh", "plot_type": "hexbin", "variable": "biomass"}

values = {"cmap": "inferno"}
values["cmap"] = values.get("cmap", "viridis")
values

HexbinPlot.create(**{"kind": "mesh", "plot_type": "hexbin", "variable": "biomass"})

self = HexbinPlot(**{"kind": "mesh", "plot_type": "hexbin", "variable": "biomass"})

# Prepare plotting parameters
validated_parameters = egp.validate_plot_args(**parameters)

self = MeshPlot(**{"kind": "mesh", "cmap": "viridis"})
self.cmap
hasattr(self, "colorbar_label")

cls = ReferenceParams

ReferenceParams.create(**{"abundance": {}})

ReferenceParams.create(**{"variable": "biomass"})


kwargs = {"kind": "mesh", "plot_type": "hexbin", "variable": "nasc"}
cls.judge(**kwargs)

values = {"dataset": plot_info.get("data"), "variable": "biomass"}

AbundanceVar(**{"dataset": plot_info.get("data"), "variable": "biomass"})

cls._REFERENCE_VAR_FACTORY(kwargs["variable"]).judge(**kwargs).model_dump(exclude_none=False)



# Plot
# plot_info.get("function")(plot_info.get("data"), **validated_parameters)
dataset = plot_info.get("data")
geo_config = validated_parameters.get("geo_config")
plot_type = validated_parameters.get("plot_type")
variable = validated_parameters.get("variable")
kwargs = validated_parameters
kwargs = {k: v for k,v in kwargs.items() 
          if k not in ["geo_config", "kind", "plot_type", "variable"]}

# Prepare units
units = "kg"
kriged_variable = "biomass"

# Get the dataset variable name
variable_col = (
    "kriged_mean"
    if variable == "biomass_density"
    else (
        "sample_cv"
        if variable == "kriged_cv"
        else "sample_variance" if variable == "local_variance" else variable
    )
)

# Get the x-axis values
x = dataset["longitude"].values
# ---- Get the y-axis values
y = dataset["latitude"].values
# ---- Get the z-axis values
z = dataset[variable_col].values

# Get axis limits
if not kwargs.get("axis_limits"):
    # ---- Get survey bounds
    survey_bounds = get_survey_bounds(dataset, geo_config)
    # ---- Additional buffering
    axis_limits = dict(
        x=dict(xmin=survey_bounds[0] * 1.005, xmax=survey_bounds[2] * 0.995),
        y=dict(ymin=survey_bounds[1] * 0.995, ymax=survey_bounds[3] * 1.005),
    )
else:
    axis_limits = kwargs.pop("axis_limits")

# Adjust the plotting properties
if variable == "biomass":
    cmap = kwargs.get("cmap", "plasma")
    reduce_C_function = kwargs.get("reduce_C_function", np.sum)
    colorbar_label = kwargs.get("colorbar_label", "Kriged biomass\n$\\mathregular{kg}$")
    vmax = kwargs.get("vmax", 10 ** np.round(np.log10(z.max())))
elif variable == "biomass_density":
    cmap = kwargs.get("cmap", "inferno")
    reduce_C_function = kwargs.get("reduce_C_function", np.mean)
    colorbar_label = kwargs.get(
        "colorbar_label", "Kriged " + kriged_variable + f" density\n{units} " + "nmi$^{-2}$"
    )
    vmax = kwargs.get("vmax", 10 ** np.round(np.log10(z.max())))
elif variable == "kriged_variance":
    cmap = kwargs.get("cmap", "hot")
    reduce_C_function = kwargs.get("reduce_C_function", np.mean)
    colorbar_label = kwargs.get(
        "colorbar_label",
        f"Kriged {kriged_variable} density variance" + f"\n({units} " + "nmi$^{-2})^{2}$",
    )
    vmax = kwargs.get("vmax", 10 ** np.round(np.log10(z.max()), 1))
elif variable == "kriged_cv":
    cmap = kwargs.get("cmap", "magma")
    reduce_C_function = kwargs.get("reduce_C_function", np.mean)
    colorbar_label = kwargs.get("colorbar_label", "Kriged $CV$")
    vmax = kwargs.get("vmax", np.ceil(z.max() / 0.1) * 0.1)
elif variable == "local_variance":
    cmap = kwargs.get("cmap", "cividis")
    reduce_C_function = kwargs.get("reduce_C_function", np.mean)
    colorbar_label = kwargs.get(
        "colorbar_label", f"Sample {kriged_variable} variance" + f"\n{units}" + "$^{-2}$"
    )
    vmax = kwargs.get("vmax", 10 ** np.round(np.log10(z.max())))

# Get vmin
vmin = kwargs.pop("vmin") if "vmin" in kwargs else 0.0

# Prepare default parameters
# ---- x
# kwargs.update(dict(xlabel=kwargs.get("xlabel", "Longitude (\u00B0E)")))
xlabel = kwargs.pop("xlabel") if "xlabel" in kwargs else "Longitude (\u00B0E)"
# ---- y
ylabel = kwargs.pop("ylabel") if "ylabel" in kwargs else "Latitude (\u00B0N)"

# Initialize figure
# ---- Update the 'figsize' if it doesn't yet exist
fig_size = kwargs.pop("figsize") if "figsize" in kwargs else apply_aspect_ratio(5.5, axis_limits)
# ---- Prune the kwargs
figure_pruned = {k: kwargs.pop(k) for k in prune_args(plt.figure, **kwargs)}
# ---- Prepare figure
plt.figure(**{**{"figsize": fig_size}, **figure_pruned})

# Initialize GeoAxes
# ---- Prune the kwargs
geoaxes_pruned = {k: kwargs.pop(k) for k in prune_args(plt.axes, **kwargs)}
# ---- Define GeoAxes
ax = plt.axes(projection=geo_config["plot_projection"], 
              **geoaxes_pruned)
# ---- Add coastline
ax.add_feature(geo_config["coastline"])

kwargs
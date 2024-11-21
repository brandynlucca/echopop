import re
from typing import Any, Callable, Dict, Literal, Optional
import numpy as np
import pandas as pd

import cartopy.feature as cfeature
from cartopy.crs import PlateCarree, Projection
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, SerializeAsAny, model_validator


class BasePlotModel(BaseModel):
    """
    Base Pydantic model for scrutinizing plotting parameters
    """

    kind: Literal["age_length_distribution", "mesh", "transect"]

    # Validator method
    @classmethod
    def judge(cls, **kwargs):
        """
        Validator method
        """
        try:
            return cls(**kwargs)
        except ValidationError as e:
            e.__traceback__ = None
            raise e


class PlotModel(BasePlotModel):
    """
    Base Pydantic model for scrutinizing plotting parameters
    """

    axis_limits: Optional[Dict[str, Any]] = Field(default=None)
    log_base: Optional[float] = Field(default=None, gt=0.0, allow_inf_nan=False)
    model_config = ConfigDict(extra="allow")

    @field_validator("axis_limits", mode="before")
    def validate_axis_limits(cls, v):

        # Check whether `v` is None or is a valid tuple
        if v is None:
            return v

        # Validate the expected dictionary format
        if not isinstance(v, dict):
            raise ValueError("Input for `axis_limits` must be a dictionary.")

        # Validate all items within the dictionary
        for key, value in v.items():
            # ---- Validate the expected 2-value format
            if set(["xmin", "xmax"]).issubset(value):
                test_values = value
            elif set(["ymin", "ymax"]).issubset(value):
                test_values = value
            elif set(["left", "right"]).issubset(value):
                test_values = value
            else:
                raise ValueError(
                    f"Input for `axis_limits['{key}']` must comprise either two values represented "
                    f"represented by '{key}min'/'{key}max' or 'left'/'right'."
                )
            # ---- Check that both values are floats
            if not all([isinstance(val, (float, int)) for _, val in test_values.items()]):
                raise ValueError(
                    f"Input for `axis_limits['{key}']` must comprise float values for limits."
                )

        # Return
        return v

    # Factory method
    @classmethod
    def create(cls, **kwargs):
        """
        Factory creation method
        """

        # Scrutinize `kind`
        cls.judge(**kwargs)

        # Find the correct sub-class plotting model
        return (
            cls._SUBCLASS_FACTORY()[kwargs["kind"]].judge(**kwargs).model_dump(exclude_none=False)
        )

    @classmethod
    def _SUBCLASS_FACTORY(cls):

        return {
            "age_length_distribution": BiologicalHeatmapPlot,
            "mesh": MeshPlot,
            "transect": TransectPlot,
        }


class GeoConfigPlot(BaseModel, arbitrary_types_allowed=True):
    """
    Geospatial parameters

    Parameters
    ----------
    init: str
        EPSG projection code.
    coastline: cartopy.feature.NaturalEarthFeature
        A land or coastline feature used for geospatial plotting.
    """

    coastline: cfeature.Feature = Field(
        default_factory=lambda: cfeature.NaturalEarthFeature(
            category="physical",
            name="land",
            scale="10m",
            facecolor="#C5C9C7",
            edgecolor="#929591",
            linewidth=0.5,
            zorder=1,
        )
    )
    init: str = Field(default="epsg:4326", description="EPSG projection code.")
    plot_projection: Projection = Field(default=PlateCarree())
    model_config = ConfigDict(extra="allow")

    @field_validator("init", mode="before")
    def validate_init(cls, v):
        # ---- Convert to a string if read in as an integer
        if isinstance(v, (int, float)):
            v = str(v)
        # ---- Convert to lowercase
        v = v.lower()
        # ---- Mold the entry into the expected format that includes a preceding 'epsg:'
        if not v.startswith("epsg"):
            v = "epsg:" + v
        # ---- Ensure that the colon is present
        if ":" not in v:
            v = "epsg:" + v.split("epsg")[1]
        # ---- Evaluate whether the pre-validator succeeded in finding an acceptable format
        if not re.match(r"^epsg:\d+$", v):
            raise ValueError(
                f"Echopop cannot parse the defined EPSG code ('{v}'). EPSG codes most be formatted "
                f"with strings beginning with 'epsg:' followed by the integer number code (e.g. "
                f"'epsg:4326')."
            )
        # ---- Return the pre-validated entry
        return v

    # Factory method
    @classmethod
    def create(cls, **kwargs):
        """
        Factory creation method
        """

        return cls(**kwargs).model_dump(exclude_none=False)


class SpatialPlot(PlotModel):
    """
    Base Pydantic model for spatial plots
    """

    geo_config: SerializeAsAny[GeoConfigPlot] = Field(
        default_factory=lambda: GeoConfigPlot.create()
    )
    model_config = ConfigDict(extra="allow")


class TransectPlot(SpatialPlot):

    plot_type: Literal["scatter"] = Field(default="scatter")
    variable: Literal[
        "abundance",
        "abundance_female",
        "abundance_male",
        "biomass",
        "biomass_density",
        "biomass_density_female",
        "biomass_density_male",
        "biomass_female",
        "biomass_male",
        "nasc",
        "number_density",
        "number_density_female",
        "number_density_male",
    ] = Field(default="biomass")

    @field_validator("plot_type", mode="before")
    def validate_plot_type(cls, v):
        if v is None:
            return "scatter"
        else:
            return v

class BaseVarRef(BaseModel):
    vmin: float = Field(default=0.0, description="Minimum value for the colormap.")
    model_config = ConfigDict(extra="allow")   
    
    # Validator method
    @classmethod
    def judge(cls, **kwargs):
        """
        Validator method
        """
        try:
            return cls(**kwargs)
        except ValidationError as e:
            e.__traceback__ = None
            raise e

class AbundanceVar(BaseVarRef, arbitrary_types_allowed=True):
    dataset: pd.DataFrame
    cmap: str = Field(default="viridis")
    variable: str
    vmax: float = Field(default_factory=AbundanceVar.vmax_gen())
    model_config = ConfigDict(extra="allow")
        
    @model_validator(mode="before")
    @classmethod
    def validate_vmax(cls, values):
        
        # Set `vmax` after `dataset` and `variable` are initialized
        if not values.get("vmax", None):
            # ---- Get the dataset column
            
            values["vmax"] = values["dataset"][values["variable"]].max()
        
        return values
    
    @classmethod
    def vmax_gen(cls, v):
        
        z = (v["dataset"][v["variable"]])
        return 10 ** np.round(np.log10(z.max()))

      

class BiomassVar(BaseVarRef):
    cmap: str = Field(default="inferno")
    vmin: float = Field(default=0.0, description="Minimum value for the colormap.")
    model_config = ConfigDict(extra="allow")

class BiomassDensityVar(BaseVarRef):
    cmap: str = Field(default="plasma")
    
    model_config = ConfigDict(extra="allow")

class NASCVar(BaseVarRef):
    cmap: str = Field(default="cividis")
    vmin: float = Field(default=0.0, description="Minimum value for the colormap.")
    model_config = ConfigDict(extra="allow")

class ReferenceParams(BaseVarRef):
    cmap: Optional[str] = Field(default=None, description="Colormap name.")
    vmax: Optional[float] = Field(default=None, description="Maximum value for the colormap.")    
    model_config = ConfigDict(extra="allow")

    # Factory method
    @classmethod
    def create(cls, **kwargs):
        """
        Factory creation method
        """

        # Scrutinize `kind`
        cls.judge(**kwargs)

        # Find the correct sub-class plotting model
        return (
            cls._REFERENCE_VAR_FACTORY(kwargs["variable"])
            .judge(**kwargs).model_dump(exclude_none=False)
        )

    @classmethod
    def _REFERENCE_VAR_FACTORY(cls, variable: str):
        
        if "abundance" in variable:
            return AbundanceVar
        elif "biomass_density" in variable:
            return BiomassDensityVar
        elif "biomass" in variable:
            return BiomassVar
        elif "nasc" in variable:
            return NASCVar
        else:
            return None
        
        
    
class ReferenceParams(BaseModel):
    cmap: Optional[str] = Field(default=None, description="Colormap name.")
    colorbar_label: Optional[str] = Field(default=None, description="Colorbar label.")
    vmax: Optional[float] = Field(default=None, description="Upper limit of the color scale.")
    vmin: Optional[float] = Field(default=None, description="Lower limit of the color scale.")
    model_config = ConfigDict(extra="allow")
    
    @model_validator(mode="before")
    @classmethod
    def reference_kwargs(cls, values):
        """
        Assign additional kwargs based on the plot type
        """
        
        # Determine if variable is sexed
        if "_male" in values["variable"]:
            label_prepend = "Male "
        elif "_female" in values["variable"]:
            label_prepend = "Female "
        else:
            label_prepend = ""

        # Set variable if missing
        # ---- Abundance
        if "abundance" in values["variable"]:
            values["cmap"] = values.get("cmap", "viridis")
        # ---- Kriged biomass
        elif "kriged_biomass_density" in values["variable"]:
            values["cmap"] = values.get("cmap", )
        # ---- Kriged biomass density
        # ---- Biomass density
        elif "biomass_density" in values["variable"]:
            values["cmap"] = values.get("cmap", "plasma")
        # ---- Biomass
        elif "biomass" in values["variable"]:            
            values["cmap"] = values.get("cmap", "inferno")
        # ---- Kriged biomass
        # ---- Kriged biomass density
        # ---- Kriged CV
        
        # ---- Biomass density
        elif "nasc" in values["variable"]:
            values["cmap"] = values.get("cmap", "cividis")  
        # ---- Number density
        elif "number_density" in values["variable"]:   
            values["cmap"] = values.get("cmap", "magma")                 
        # Return values
        return values
        
    # Factory method    
    @classmethod
    def create(cls, **kwargs):
        """
        Factory creation method
        """

        return cls(**kwargs).model_dump(exclude_none=False)
    
class HexbinPlot(ReferenceParams):
    reduce_C_function: Optional[Callable] = Field(
        default=None, 
        description="Function to use for reducing the C-dimension."
    )
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def reference_kwargs(cls, values):
        """
        Assign additional kwargs based on the plot type
        """

        # Set variable if missing
        input = values.get("reduce_C_function", None)
        # ---- Assign if missing
        if input is None:
            # ---- Biomass
            if values["variable"] == "biomass":            
                values["reduce_C_function"] = np.sum
            # ---- All other variables
            else:
                values["reduce_C_function"] = np.mean
                
        # Return values
        return values


    
           


class MeshPlot(SpatialPlot):

    plot_type: Optional[Literal["hexbin", "pcolormesh", "scatter"]] = Field(default="hexbin")
    variable: Literal[
        "biomass", "biomass_density", "kriged_variance", "kriged_cv", "local_variance"
    ] = Field(default="biomass")

    @field_validator("plot_type", mode="before")
    def validate_plot_type(cls, v):
        if v is None:
            return "hexbin"
        else:
            return v
        
    @model_validator(mode="after")
    def reference_kwargs(self):        
        """
        Assign additional kwargs based on the plot type
        """
        
        # Biomass
        if self.variable == "biomass":
            # ---- Generics
            self.cmap = "plasma"
            self.colorbar_label = "Kriged biomass\n$\\mathregular{kg}$"
            



class BiologicalHeatmapPlot(PlotModel):
    """
    Base Pydantic model for biological heatmap plots
    """

    grid_heatmap: bool = Field(default=True)
    plot_type: Literal["heatmap"] = Field(default="heatmap")
    sex: Literal["all", "female", "male"] = Field(default="all")
    variable: Literal["abundance", "biomass"] = Field(default="biomass")

    @field_validator("plot_type", mode="before")
    def validate_plot_type(cls, v):
        if v is None:
            return "heatmap"
        else:
            return v

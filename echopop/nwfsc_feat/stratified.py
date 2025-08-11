import geopy.distance
import pandas as pd
import numpy as np
import awkward as awk

from typing import Any, Dict, List, Optional, Tuple
from . import load_data, utils


class JollyHampton:
    """
    """

    def __init__(
        self,
        model_parameters: Dict[str, Any],
        resample_seed: Optional[int] = None,
    ):

        # Ingest model parameters
        self.model_params = model_parameters

        # Initialize the random number generator
        self.rng = np.random.default_rng(resample_seed)

        # Initialize attributes
        self.resampled_df = None 

    def _partition_data_into_transects(
        self,
        data_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partition the gridded dataset into virtual transects based on latitude
        """

        # Get model parameters
        mp = self.model_params

        # Partition the dataset based on latitude
        data_df.loc[:, "latitude"] = (
            np.round(data_df.loc[:, "latitude"] * mp["transects_per_latitude"] + 0.5) /
            mp["transects_per_latitude"]
        )

        # Create unique key pairs for latitude and transect
        unique_latitude_transect_key = pd.DataFrame(
            {
                "latitude": np.unique(data_df["latitude"]),
                "transect_num": np.arange(0, len(np.unique(data_df["latitude"])), 1) + 1,
            }
        ).set_index("latitude")

        # Temporarily set index
        data_df.set_index("latitude", inplace=True)

        # Append the transect numbers
        data_df["transect_num"] = unique_latitude_transect_key

        # Return the partitioned gridedd dataset and unique latitude-transect key
        return data_df.reset_index(), unique_latitude_transect_key.reset_index()
        

    def _initialize_virtual_df(
        self,
        data_df: pd.DataFrame,
        variable: str,
    ) -> pd.DataFrame:
        """
        Create virtual transects from gridded data 
        """

        return data_df[
            ["transect_num", "longitude", "latitude", "area", variable]
        ].rename(
            columns={"area": "area_interval"}
        ).sort_values(["transect_num"]).set_index(["transect_num"])

    def _create_virtual_df(
        self,
        virtual_df: pd.DataFrame,
        latitude_key: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the transect distances and areas
        """

        # Initialize with the latitude values
        virtual_transect_data = virtual_df.groupby(level=0).apply(
            lambda x: x.latitude.mean()
        ).to_frame("latitude")

        # Compute the transect distances
        virtual_transect_data["distance"] = virtual_df.groupby(level=0).apply(
            lambda x: geopy.distance.distance(
                (x.latitude.min(), x.longitude.min()), (x.latitude.max(), x.longitude.max())
            ).nm
        )

        # Compute the areas
        virtual_transect_data.loc[:, "area"] = np.where(
            virtual_transect_data.index.isin([
                virtual_df.index.min(), virtual_df.index.max()
            ]),
            virtual_transect_data["distance"] * np.diff(
                latitude_key["latitude"]
            ).mean() * 60,
            virtual_transect_data["distance"] * np.diff(
                latitude_key["latitude"]
            ).mean() * 60 / 2
        )

        # Get variable column name
        variable_col = list(
            set(virtual_df.columns).difference(["longitude", "latitude", "area_interval"])
        )

        # Sum the variable
        virtual_transect_data[variable_col] = virtual_df.groupby(level=0)[variable_col].sum()
        
        # Return the summarized transect data
        return virtual_transect_data

    def create_virtual_transects(
        self,
        data_df: pd.DataFrame,
        geostrata_df: pd.DataFrame,
        stratify_by: List[str],
        variable: str
    ):

        # Assign transect numbers and get the latitude-transect key
        data_proc, latitude_key = self._partition_data_into_transects(
            data_df.copy()
        )

        # Initialize the virtual transects
        transects_df = self._initialize_virtual_df(
            data_proc, variable
        )

        # Compute the summary metrics
        virtual_df = self._create_virtual_df(
            transects_df, latitude_key
        )

        # Stratify the virtual transects
        virtual_df = load_data.join_geostrata_by_latitude(
            virtual_df, 
            geostrata_df, 
            stratum_name=stratify_by[0]
        )

        # Return the new virtual transects
        return virtual_df

    def _prepare_bootstrap_arrays(
        self,
        data_df: pd.DataFrame,
        stratum_index: pd.DataFrame,
        num_transects_to_sample: pd.DataFrame,
        variable: str,
    ) -> Tuple[awk.Array, awk.Array, awk.Array]:

        # For each stratum, generate replicate samples of transect indices
        transect_samples = [
            np.sort(
                np.array([
                    self.rng.choice(stratum_index.loc[j, "transect_num"].values, 
                            size=num_transects_to_sample.loc[j], replace=False)
                    for _ in range(self.model_params["num_replicates"])
                ])
            )
            for j in num_transects_to_sample.index
        ]

        # Use advanced indexing to get the sampled values
        # ---- Distances
        sampled_distances = awk.Array([
            [data_df.loc[replicate, "distance"].to_numpy() for replicate in stratum]
            for stratum in transect_samples
        ])
        # ---- Areas
        sampled_areas = awk.Array([
            [data_df.loc[replicate, "area"].to_numpy() for replicate in stratum]
            for stratum in transect_samples
        ])
        # ---- Variable
        sampled_values = awk.Array([
            [data_df.loc[replicate, variable].to_numpy() for replicate in stratum]
            for stratum in transect_samples
        ])

        # Return a tuple of the uneven arrays
        return sampled_distances, sampled_areas, sampled_values     

    @staticmethod
    def _get_variance(
        values: awk.Array,
        value_mean: awk.Array,
        weights: awk.Array,
        dof: np.array,
    ) -> awk.Array:

        # Calculate the squared deviation
        squared_deviation = (values - value_mean[..., None]) ** 2

        # Sum the weighted squared deviations
        squared_deviation_wgt = awk.sum(weights**2 * squared_deviation, axis=-1)

        # Return the variance
        return squared_deviation_wgt / awk.Array(dof.values)[:, None]

    def stratified_bootstrap(
        self,
        data_df: pd.DataFrame,
        stratify_by: List[str],
        variable: str,
    ):

        # Get the model parameters
        mp = self.model_params

        # Enumerate the number of transects per stratum
        strata_transect_counts = data_df.groupby(stratify_by, observed=False)["distance"].count()

        # Calculate the total transect area per stratum
        strata_transect_areas = data_df.groupby(stratify_by, observed=False)["area"].sum()

        # Index by stratum
        stratum_index = data_df[stratify_by].reset_index().set_index(stratify_by)

        # Calculate the number of transects per stratum
        num_transects_to_sample = utils.roundn(
            strata_transect_counts * mp["strata_transect_proportion"]
        ).astype(int)

        # Offset term used for later variance calculation
        sample_offset = np.where(num_transects_to_sample == 1, 0, 1)

        # Calculate effective sample size/degrees of freedom for variance calculation
        sample_dof = num_transects_to_sample * (num_transects_to_sample - sample_offset)

        # Compute the resampling arrays
        sampled_distances, sampled_areas, sampled_values = self._prepare_bootstrap_arrays(
            data_df, stratum_index, num_transects_to_sample, variable
        )

        # Compute the stratified weights
        stratified_weights = sampled_distances / awk.mean(sampled_distances, axis=-1, keepdims=True)

        # Adjust the values based on distance
        values_adjusted = sampled_values / sampled_distances

        # Compute the value mean
        mean_arr = awk.sum(
            sampled_values * sampled_distances, axis=-1
        ) / awk.sum(sampled_distances, axis=-1)

        # Calculate the variance
        variance_arr = self._get_variance(
            values_adjusted, mean_arr, stratified_weights, sample_dof
        )

        # Calculate the squared deviation
        squared_deviation = (values_adjusted - mean_arr[..., None]) ** 2

        # Sum the weighted squared deviations
        squared_deviation_wgt = awk.sum(stratified_weights**2 * squared_deviation, axis=-1)

        # Return the variance
        variance_arr = squared_deviation_wgt / awk.Array(sample_dof.values)[:, None]

        # Convert and transpose all arrays in one step
        length_arr_np, mean_arr_np, area_arr_np, total_arr_np, variance_arr_np = [
            awk.to_numpy(arr).T
            for arr in [awk.sum(sampled_distances, axis=-1), 
                        mean_arr, 
                        awk.sum(sampled_areas, axis=-1), 
                        awk.sum(sampled_values, axis=-1), 
                        variance_arr]
        ]

        pd.DataFrame(mean_arr_np)
         pd.DataFrame(mean_arr_np, columns=[f'col{i+1}' for i in range(mean_arr_np.shape[1])])
        df = pd.DataFrame(mean_arr_np, columns=range(1, mean_arr_np.shape[1] + 1))

        # Stack all values into a single Series indexed by column number repeated
        series = df.melt(var_name='col', value_name='value').set_index('col')['value']

        # Create DataFrame
        pd.DataFrame({
            stratify_by[0]: np.repeat(strata_transect_counts.index,
                                      mp["num_replicates"]),
            "realization": np.tile(np.arange(1, mp["num_replicates"] + 1), 
                                   len(strata_transect_counts)),
        }).set_index(stratify_by)        

        ((total_arr_np * strata_transect_areas.to_numpy()).sum(axis=1) / strata_transect_areas.to_numpy().sum())[0] * 1e-8

        np.sqrt((variance_arr_np * strata_transect_areas.to_numpy()**2).sum(axis=1))[0]/(mean_arr_np*strata_transect_areas.to_numpy()).sum(axis=1)[0]

        # Compute the straum densities
        stratum_density = mean_arr_np / length_arr_np

        mean_arr_np.sum(axis=1)

        (stratum_density * strata_transect_areas.to_numpy()).sum(axis=1)

        # Compute the stratum totals
        stratum_total = stratum_density * area_arr_np

        stratum_total.sum(axis=1) * 1e-6
        unweighted_stratum_total.sum(axis=1) * 1e-6
        # Update `resampled_df`
        pd.DataFrame(
            {
                "realization": np.arange(1, mp["num_replicates"] + 1),
            }
        )

        # ---- By stratum (density)
        unweighted_stratum_density = mean_arr_np / length_arr_np
        (unweighted_stratum_density * strata_transect_areas.to_numpy()).sum(axis=1)
        unweighted_stratum_total.sum(axis=1)
        # ---- By stratum (total)
        unweighted_stratum_total = unweighted_stratum_density * strata_transect_areas.to_numpy()
        # unweighted_stratum_total = unweighted_stratum_density *area_arr_np

        # ---- By survey (total)
        unweighted_survey_total = unweighted_stratum_total.sum(axis=1)

        # ---- By survey (density)
        unweighted_survey_density = unweighted_survey_total / strata_transect_areas.sum()

        # ---- Proportional stratum distributions
        unweighted_stratum_proportions = total_arr_np / total_arr_np.sum(axis=1, keepdims=True)

        # ---- Transect-length weighted coefficient of variation (CV)
        weighted_variance = (variance_arr_np * strata_transect_areas.to_numpy()**2).sum(axis=1)
        # weighted_variance = (variance_arr_np * area_arr_np**2).sum(axis=1)

        # ---- Convert to the standard deviation
        weighted_stdev = np.sqrt(weighted_variance)
        weighted_mean = (mean_arr_np * strata_transect_areas.to_numpy()).sum(axis=1)
        # weighted_mean = (mean_arr_np * area_arr_np).sum(axis=1)
        bootstrap_cv = weighted_stdev / weighted_mean
        weighted_stdev.min() * 1e-10; weighted_stdev.max() * 1e-10
        bootstrap_cv.min(); bootstrap_cv.mean(); bootstrap_cv.max()

        A = (total_arr_np * strata_transect_areas.to_numpy()).sum(axis=1)/strata_transect_areas.to_numpy().sum()
        A.min() * 1e-8; A.mean() * 1e-8; A.max() * 1e-8
        
        
        


        # Sum the quantities over each stratum
        # ---- Transect lengths
        distance_totals = awk.sum(sampled_distances, axis=-1)
        # ---- Areas
        area_totals = awk.sum(sampled_areas, axis=-1)
        # ---- Biological variable
        variable_totals = awk.sum(sampled_values, axis=-1)


        

        # Output the related summary statistics
        # ---- Save the output resampled distributions
        resampled_distributions = pd.DataFrame(
            {
                "realization": np.arange(1, transect_replicates + 1),
                "unweighted_survey_density": unweighted_survey_density,
                "unweighted_survey_total": unweighted_survey_total,
                "weighted_survey_total": weighted_mean,
                "weighted_survey_variance": weighted_variance,
                "survey_cv": bootstrap_cv,
            }
        )

        # Sum the quantities over each stratum
        # ---- Transect lengths
        distance_totals = awk.sum(sampled_distances, axis=-1)
        # ---- Areas
        area_totals = awk.sum(sampled_areas, axis=-1)
        # ---- Biological variable
        variable_totals = awk.sum(sampled_values, axis=-1)

        

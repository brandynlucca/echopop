from typing import Union, Optional
from pathlib import Path
import copy

from .live_core import(
    LIVE_DATA_STRUCTURE,
)

from ..acoustics import (
    ts_length_regression,
    to_dB,
    to_linear
)

from .sql_methods import query_processed_files
from .live_acoustics import preprocess_acoustic_data, compute_nasc
from .live_biology import preprocess_biology_data


from . import live_data_processing as eldp
from . import live_data_loading as eldl
class LiveSurvey:
    """
    A real-time processing version of the `echopop` base `Survey` class that ingests biological, 
    acoustic, and event meta data to provide population estimates when generated.
    """

    def __init__(
        self,
        live_init_config_path: Union[str, Path], 
        live_file_config_path: Union[str, Path],
        verbose: bool = True,
    ):
        # Initialize `meta` attribute
        self.meta = copy.deepcopy(LIVE_DATA_STRUCTURE["meta"])

        # Loading the configuration settings and definitions that are used to
        # initialize the Survey class object
        self.config = eldl.live_configuration(Path(live_init_config_path), 
                                              Path(live_file_config_path))
        # ---- Initialize config key for database files
        self.config.update(
            {"database": {key: None for key in self.config["input_directories"].keys()}}
        )
        
        # Initialize input attribute
        self.input = copy.deepcopy(LIVE_DATA_STRUCTURE["input"])

        # Initialize database attribute
        self.database = copy.deepcopy(LIVE_DATA_STRUCTURE["database"])

        # Initialize the results attribute
        self.results = copy.deepcopy(LIVE_DATA_STRUCTURE["results"])

        # Configure the spatial settings
        self.input.update({"spatial": eldl.configure_spatial_settings(self.config)})

        # TODO: Add verbosity for printing database filepaths/connections 
        if verbose: 
            pass

    def load_acoustic_data(self,
                           input_filenames: Optional[list] = None,
                           verbose: bool = True):
        
        # Validate the data directory and format the filepaths
        acoustic_files = eldl.validate_data_directory(self.config, dataset="acoustics", 
                                                      input_filenames=input_filenames)
        
        # Read in the acoustic data files
        if acoustic_files:
            # ! [REQUIRES DASK] ---- Read in the listed file
            # ---- Read in the acoustic data files
            prc_nasc_df, acoustic_data_units = eldl.read_acoustic_files(acoustic_files)
            # ---- Add the `acoustic_data_units` to the dictionary
            self.config["acoustics"]["dataset_units"] = acoustic_data_units   
            # ---- Preprocess the acoustic dataset
            # TODO: SettingWithCopyWarning:
            self.input["acoustics"]["prc_nasc_df"] = preprocess_acoustic_data(prc_nasc_df.copy(), 
                                                                              self.input["spatial"],
                                                                              self.config)     
            # TODO: Add verbosity for printing database filepaths/connections 
            if verbose:
                # ---- Create file list
                file_list = "\n".join(acoustic_files)
                print(
                    f"The following acoustic files have been processed:\n"
                    f"{file_list}."
                )
        else:
            self.input["acoustics"]["prc_nasc_df"] = None

    def load_biology_data(self,
                          input_filenames: Optional[list] = None,
                          verbose: bool = True):

        # Validate the data directory and format the filepaths
        biology_files = eldl.validate_data_directory(self.config, dataset="biology", 
                                                     input_filenames=input_filenames)
        
        # TODO: Add verbosity for printing database filepaths/connections 
        if biology_files and verbose:
            # ---- Create file list
            file_list = "\n".join(biology_files)
            print(  
                f"The following biological files have been processed:\n"
                f"{file_list}."
            )
        
        # Read in the biology data files
        initial_biology_output = eldl.read_biology_files(biology_files, self.config)

        # Preprocess the biology dataset
        self.input["biology"], self.input["biology_processed"] = (
            preprocess_biology_data(initial_biology_output, self.input["spatial"], self.config)
        )

    def process_biology_data(self):
        # method here
        pass
    
    def process_acoustic_data(self, echometrics: bool = True):
        
        # Get the unprocessed acoustic data
        acoustic_data_df = self.input["acoustics"]["prc_nasc_df"]

        # Integrate NASC (and compute the echometrics, if necessary)
        nasc_data_df = compute_nasc(acoustic_data_df, self.config, echometrics)
        
        # Format the dataframe and insert into the LiveSurvey object
        self.input["nasc_df"] = nasc_data_df
    
    def estimate_population(self):
        # method here
        pass 
        
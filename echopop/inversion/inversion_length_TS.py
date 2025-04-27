import pandas as pd
from .inversion_base import InversionBase


class InversionLengthTS(InversionBase):
    """
    Class to perform inversion using length-TS regression.
    """

    def __init__(self, df_model_params: pd.DataFrame):
        super().__init__(df_model_params)
        
        # Set inversion method
        self.inversion_method = "length_TS_regression"

        # Check df_model_params
        # -- check if df_model_params contain all required parameters
        # -- for length-TS regression these are slope and intercept        

    def invert(self, df_nasc: pd.DataFrame) -> pd.DataFrame:

        # Make df_model_params easy to combine

        # Join df_model_params and df_nasc

        # Perform inversion (regression)

        pass

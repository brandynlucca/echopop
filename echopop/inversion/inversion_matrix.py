import pandas as pd
from .inversion_base import InversionBase


class InversionMatrix(InversionBase):

    def __init__(self, df_model_params: pd.DataFrame):
        super().__init__(df_model_params)
        
        # Set inversion method
        self.inversion_method = "krill_matrix_inversion"

        # Check df_model_params
        # -- check if df_model_params contain all required parameters
        # -- for matrix inversion these are the krill shape parameters
        
    def invert(self, df_nasc: pd.DataFrame) -> pd.DataFrame:

        # Krill inversion ops
        # -- note any non-stratum grouping should be performed OUTSIDE of this class

        pass

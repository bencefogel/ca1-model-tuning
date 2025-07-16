import pandas as pd
import gc

from preprocessor.utils.preprocess_intrinsic import preprocess_intrinsic
from preprocessor.utils.preprocess_synaptic import preprocess_synaptic


class MembraneCurrentPreprocessor:
    """
    Preprocesses intrinsic and synaptic currents and combines them into membrane currents.
    """
    def __init__(self) -> None:
        """
        Initializes the MembraneCurrentPreprocessor with an empty DataFrame.

        Args:
            membrane_currents_combined (pd.DataFrame): A DataFrame containing combined membrane currents.
        """
        self.membrane_currents_combined = pd.DataFrame()

    def combine_membrane_currents(self, simulation_data: dict) -> None:
        """
        Combines intrinsic and synaptic currents into a single DataFrame.

        This method preprocesses intrinsic and synaptic currents using utility functions,
        concatenates the resulting DataFrames, and sets the processed data to the
        'membrane_currents_combined' attribute.

        Args:
            simulation_data (dict): The simulation data containing 'intrinsic_data', 'synaptic_data',
                                    and 'areas' used for preprocessing.

        Example of expected simulation_data structure:
            simulation_data = {
                'intrinsic_data': ([segments], [values]),
                'synaptic_data': ([segments], [values]),
                'areas': [area_values]
            }
        """
        isegments = simulation_data['intrinsic_data'][0]
        ivalues = simulation_data['intrinsic_data'][1]
        area = simulation_data['areas']
        ssegments = simulation_data['synaptic_data'][0]
        svalues = simulation_data['synaptic_data'][1]

        intrinsic = preprocess_intrinsic(isegments, ivalues, area)
        synaptic = preprocess_synaptic(ssegments, svalues)

        # Create merged dataframe
        dfs = intrinsic + synaptic
        del intrinsic, synaptic
        gc.collect()

        df_im = pd.concat(dfs)
        del dfs
        gc.collect()

        df_im['index'] = df_im['index'].astype('category')
        df_im['itype'] = df_im['itype'].astype('category')

        # Calculate and set multiindex
        segments = df_im['index'].unique()
        itypes = df_im['itype'].unique()

        multi_index = pd.MultiIndex.from_product([segments, itypes], names=['segment', 'itype'])
        df_im_combined = df_im.set_index(['index', 'itype']).reindex(multi_index)
        del df_im
        gc.collect()

        df_im_combined = df_im_combined.fillna(0)
        df_im_combined.columns = df_im_combined.columns.astype(int)
        self.membrane_currents_combined = df_im_combined


    def merge_section_im(self, target: str) -> pd.DataFrame:
        """
        Merges membrane currents of the target section.

        This method filters membrane currents that belong to the specified target,
        aggregates them by current type, and returns a merged DataFrame.

        Args:
            target (str): The target section for merging currents.

        Returns:
            pd.DataFrame: A DataFrame with merged membrane currents for the target section.
        """
        df_section = self.membrane_currents_combined[self.membrane_currents_combined.index.get_level_values(0).str.startswith(
            f'{target}(')]  # select all rows belonging to the given segment
        df_summed_by_itype = df_section.groupby(level='itype', observed=False).sum()  # sum dataframe by current type for each time point
        df_summed_by_itype = df_summed_by_itype.reset_index()
        df_summed_by_itype['segment'] = target
        df_summed_by_itype = df_summed_by_itype.set_index(['segment', 'itype'])

        # Update original dataframe with the merged dendritic segment
        df_merged= pd.concat([self.membrane_currents_combined.drop(df_section.index), df_summed_by_itype], axis=0)
        return df_merged


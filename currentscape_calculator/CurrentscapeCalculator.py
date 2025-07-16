import pandas as pd
import numpy as np

from currentscape_calculator.partitioning_algorithm import partition_iax

class CurrentscapeCalculator:
    """
    Represents a calculator for performing Currentscape analysis for input data.
    This class is designed to calculate positive and negative membrane current components of axial currents.

    Attributes:
        target (str): The specific target for processing the data.
        partitioning_strategy (str): Strategy for partitioning the data, either by "type" or "region".
        The directory path containing .txt files, where each file corresponds to neuronal sections
        that belong to the same region.
    """
    def __init__(self, target: str, partitioning_strategy: str, regions_list_directory: str) -> None:
        self.target = target
        self.partitioning_strategy = partitioning_strategy
        self.regions_list_directory = regions_list_directory

    def calculate_currentscape(self, iax: str, im: str, taxis: np.array, tmin: int, tmax: int):
        """
            This function processes the input dataframes containing axial currents (iax)
            and membrane currents (im), then computes and partitions the
            currents according to the specified partitioning strategy.
            If no specific time interval is defined (tmin and tmax are not provided),
            partitioning is applied across the entire dataframe.

            Args:
                iax (str): Path to Dataframe containing axial current data, read
                    from a CSV file indexed by multiindex (0, 1) with integer-type labeled columns.
                im (str): Path to Dataframe containing membrane current data, read
                    from a CSV file indexed by multiindex (0, 1) with integer-type labeled columns.
                taxis (np.array): Array containing time values corresponding to the currents data.
                tmin (int): Minimum time value for the selected time interval.
                tmax (int): Maximum time value for the selected time interval.

            Returns:
                Tuple: Contains two partitioned portions of extracellular currents,
                (im_part_pos, im_part_neg), based on the specified partitioning strategy.
        """

        print("Calculating currentscape...")
        # Load data for the given pair of files
        df_iax = pd.read_csv(iax, index_col=[0,1])
        df_iax.columns = df_iax.columns.astype(int)
        df_im = pd.read_csv(im, index_col=[0,1])
        df_im.columns = df_im.columns.astype(int)

        if self.partitioning_strategy == 'type':
            df_im.sort_index(axis=0, level=(0, 1), inplace=True)

        # if no timepoints is selected, we perform the partitioning on the whole dataframe
        if tmin and tmax is None:
            t_max = df_im.shape[1]
            segment_indexes = list(range(t_max))
        else:
            segment_indexes = np.flatnonzero((taxis > tmin) & (taxis < tmax))

        # Perform the partitioning
        im_part_pos, im_part_neg = partition_iax(df_im, df_iax, timepoints=segment_indexes, target=self.target,
                                                 partition_by=self.partitioning_strategy,
                                                 regions_list_directory=self.regions_list_directory)

        im_part_pos.columns = segment_indexes
        im_part_neg.columns = segment_indexes
        return im_part_pos, im_part_neg

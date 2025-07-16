import numpy as np
import pandas as pd


def change_unit_na(currents: pd.DataFrame, area: pd.DataFrame) -> pd.DataFrame:
    """
    Convert membrane currents to nA from mA/cm2.

    Parameters:
        currents (df): DataFrame containing membrane currents.
        area (df): DataFrame containing segment areas.

    Returns
        df_converted (df): DataFrame containing membrane currents in nA.
    """
    array_converted = np.zeros_like(currents.values)
    for i, segment in enumerate(currents.index):
        segment_area = area.loc[segment].values[0]
        array_na = currents.loc[segment] * segment_area * 0.01
        array_converted[i, :] = array_na

    df_converted = pd.DataFrame(data=array_converted, index=list(currents.index), columns=list(currents.columns))
    df_converted = df_converted.reset_index()
    return df_converted


def preprocess_intrinsic(segments, values, area):
    currents = list(segments.keys())
    dfs = []
    for curr in currents:
        seg = segments[curr]
        val = values[curr]
        df = pd.DataFrame(data=val, index=seg)
        df_converted = change_unit_na(df, area)
        df_converted.insert(1, 'itype', curr)
        df_converted[['index', 'itype']] = df_converted[['index', 'itype']].astype('category')
        dfs.append(df_converted)
    return dfs

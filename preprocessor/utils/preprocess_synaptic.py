import pandas as pd


def preprocess_synaptic(segments, values):
    currents = list(segments.keys())
    dfs = []
    for curr in currents:
        seg = segments[curr]
        val = values[curr]
        df = pd.DataFrame(data=val, index=seg)
        df = df.reset_index()
        df_summed = df.groupby('index', as_index=False).sum()
        df_summed.insert(1, 'itype', curr)
        df_summed[['index', 'itype']] = df_summed[['index', 'itype']].astype('category')
        dfs.append(df_summed)
    return dfs

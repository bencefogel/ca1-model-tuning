import pandas as pd
from neuron import h


def get_segment_areas():
    segments = []
    areas = []
    for sec in h.allsec():
        for seg in sec.allseg():
            segments.append(str(seg))
            areas.append(seg.area())
    return pd.DataFrame({'area': areas}, index=segments)


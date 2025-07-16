import numpy as np
import pandas as pd
import altair as alt

from altair import VConcatChart


def get_total_pos(df: pd.DataFrame) -> pd.Series:
    return df.sum(axis=0)


def get_total_neg(df: pd.DataFrame) -> pd.Series:
    return df.sum(axis=0)


def get_cnorm_pos(df: pd.DataFrame) -> pd.DataFrame:
    total = get_total_pos(df)
    return df.div(total, axis=1)


def get_cnorm_neg(df: pd.DataFrame) -> pd.DataFrame:
    total = get_total_neg(df)
    return df.div(total, axis=1)

def create_vm_chart(v: np.ndarray, t: np.ndarray, vmin=-68, vmax=-63) -> alt.Chart:
    # Create a DataFrame with the time and potential data
    df = pd.DataFrame({
        'Time': t,
        'Vm': v
    })

    # Create a line chart using Altair
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Time:Q', title=None, axis=alt.Axis(ticks=False, grid=False, labels=False), scale=alt.Scale(domain=[np.min(t), np.max(t)])),
        y=alt.Y('Vm:Q', title='V (mV)', axis=alt.Axis(grid=False, labelFontSize=18, titleFontSize=18, titleFontWeight='normal'), scale=alt.Scale(domain=[vmin, vmax]))
    ).properties(
        width=1000,
        height=200, 
    )
    return chart


def create_currsum_pos_chart(df: pd.DataFrame, t: np.ndarray) -> alt.Chart:
    total = get_total_pos(df)
    imin = 10**(np.floor(np.log10(np.min(total))))
    ivector = np.ones(len(t)) * imin
    df_total = pd.DataFrame({'Current': total, 'ibase': ivector, 'Time': t})
    if (np.min(total) < 0.1):
        Iticks = [0.001, 0.01, 0.1, 1]
        Idomain = [0.001, 1]
    else:
        Iticks = [0.1, 1, 10, 100]
        Idomain = [0.1, 100]

    currsum_chart = alt.Chart(df_total).mark_area(
        color='black',
        opacity=0.6
    ).encode(
        x=alt.X('Time:Q', title=None, axis=alt.Axis(ticks=False, grid=False, labels=False), scale=alt.Scale(domain=[np.min(t), np.max(t)])),
        y=alt.Y('Current:Q', title='I (nA)', axis=alt.Axis(grid=True, tickCount=4, tickExtra=False, labelFontSize=18, titleFontSize=18, titleFontWeight='normal', values=Iticks), scale=alt.Scale(domain=Idomain, type="log")),#.scale(type="log")
        y2=alt.Y2('ibase:Q')#.scale(type="log")
    ).properties(
        width=1000,
        height=100
    )
    return currsum_chart


def create_currshares_chart(pos: pd.DataFrame, neg: pd.DataFrame, t: np.ndarray, partitionby='type') -> tuple[alt.Chart, alt.Chart]:
    cnorm_pos = get_cnorm_pos(pos)
    cnorm_neg = -1 * get_cnorm_neg(neg)

    if (partitionby == 'region'):
        custom_color_mapping = {
            'distal_intrinsic': '#23125d',
            'distal_synaptic': '#65559a',
            'oblique_trunk_intrinsic': '#831b6d',
            'oblique_trunk_synaptic': '#b875bb',
            'basal_intrinsic': '#c21f9a',
            'basal_synaptic': '#f0a6c1',
            'soma_intrinsic': '#ef3870',
            'soma_synaptic': '#fac4b8',
            'axon_intrinsic': '#fbe38e',
            'axon_synaptic': '#fdf3d1'
        }
    else:
        custom_color_mapping = {
            'kap': '#51a7f9',
            'kad': '#0365c0',
            'kdr': '#164f86',
            'kslow': '#002452',
            'nad': '#ec5d57',
            'nax': '#c82506',
            'car': '#f39019',
            'passive': '#00882b',
            'capacitive': '#70bf41',
            'AMPA': '#f5d328',
            'NMDA': '#c3971a',
            'GABA': '#b36ae2',
            'GABA_B': '#773f9b',
            'soma_iax_neg': '#a6aaa9',
            'soma_iax_pos': '#a6aaa9',
        }
        
    # Prepare the dataframes for positive currents
    df_cnorm_pos = pd.DataFrame(cnorm_pos.T * 100)
    df_cnorm_pos['Time'] = t
    df_cnorm_pos_long = df_cnorm_pos.melt(
        id_vars=['Time'],
        var_name='itype',
        value_name='Current'
    )

    # Prepare the dataframes for negative currents
    df_cnorm_neg = pd.DataFrame(cnorm_neg.T * 100)
    df_cnorm_neg['Time'] = t
    df_cnorm_neg_long = df_cnorm_neg.melt(
        id_vars=['Time'],
        var_name='itype',
        value_name='Current'
    )

    # Define custom color scale for the "itype" variable
    color_scale = alt.Scale(domain=list(custom_color_mapping.keys()), range=list(custom_color_mapping.values()))

    # Define the shared y-axis scale
    shared_y_scale = alt.Scale(domain=[-100, 100], padding=0)

    # Create the positive current chart
    currshares_pos_chart = alt.Chart(df_cnorm_pos_long).mark_area().encode(
        x=alt.X('Time:Q', axis=alt.Axis(grid=False, labels=True, title="time (ms)", titleFontSize = 18, titleFontWeight='normal', format='.0f'), scale=alt.Scale(domain=[np.min(t), np.max(t)])),
        y=alt.Y('Current:Q', title="inward % \t\t \t\t outward %", axis=alt.Axis(labels=False, grid=False, titleFontSize=18, titleFontWeight='normal'), scale=alt.Scale(domain=[-100, 0])),
        color=alt.Color('itype:N', scale=color_scale, legend=alt.Legend(labelFontSize=18, titleFontSize=18, titleFontWeight='normal'))
    ).properties(
        width=1000,
        height=400
    )

    # Create the negative current chart
    currshares_neg_chart = alt.Chart(df_cnorm_neg_long).mark_area().encode(
        x=alt.X('Time:Q', axis=alt.Axis(grid=False, labels=True, labelFontSize=18), scale=alt.Scale(domain=[np.min(t), np.max(t)])),
        y=alt.Y('Current:Q', axis=alt.Axis(labels=False, ticks=False, grid=False), scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('itype:N', scale=color_scale)
    ).properties(
        width=1000,
        height=400
    )
    line = alt.Chart().mark_rule().encode(y=alt.datum(0))
    
    return currshares_pos_chart, currshares_neg_chart + line

def combine_charts(vm: alt.Chart, totalpos: alt.Chart, currshares_pos: alt.Chart, currshares_neg: alt.Chart) -> VConcatChart:
    currshares = alt.layer(
        currshares_pos,
        currshares_neg
    )

    chart = alt.vconcat(totalpos, currshares).properties(
        spacing=0
    )

    chart = alt.vconcat(vm, chart).properties(
        spacing=0
    ).configure_view(
    stroke=None
    )
    return chart

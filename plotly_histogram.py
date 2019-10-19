import plotly.graph_objs as go

import numpy as np

def return_go_hist(data, start=0, end = 100, size = 30, histnorm='',
                   cumulative = False,
                   title='',
                   x_label='',
                   y_label=''):
    fig = go.Figure()
    map_dict = {'':'Count', 'percent' : 'Percentage', 'probability density' : 'probability density'}
    fig.add_trace(go.Histogram(
        x=data,
        histnorm=histnorm,
        name='text lengths', # name used in legend and hover labels
        xbins=dict( # bins used for histogram
            start=start,
            end=end,
            size=size
        ),
        marker_color='gray',
        opacity=0.6,
        cumulative_enabled = cumulative
    ))
    tickvals = list(range(int(start), int(end)+int(size), int(size)))
    ticktext = list(map(str, tickvals))

    fig.layout.update(
        title_text=title,  # title of plot
        xaxis_title_text=x_label,  # xaxis label
        yaxis_title_text='{} {}'.format(y_label, map_dict[histnorm]),  # yaxis label
        #bargap=0.2,  # gap between bars of adjacent location coordinates
        #bargroupgap=0.1,  # gap between bars of the same location coordinates
        xaxis=dict(showgrid=True, zeroline=True, gridwidth=0.1, gridcolor='black',
                   ticktext =ticktext, tickvals=tickvals),
        yaxis=dict(showgrid=True, zeroline=True, gridwidth=0.1, gridcolor='black'),
        xaxis_title_font=dict(size=15, family='Courier', color='black'),
        yaxis_title_font=dict(size=15, family='Courier', color='black'),
        title_font=dict(size=20, family='Courier', color='black')
    )

    return fig

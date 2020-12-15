import dash_core_components as dcc
import dash_html_components as html

# from dash.dependencies import Input, Output

import webapp.callbacks  # noqa: F401
from webapp.app import app
from webapp.layouts import parameters_layout, results_table_layout, results_graph_layout


app.layout = html.Div([
    html.H1(children='ClusterLogs'),

    html.Div(id='results-storage', children=None, style={'display': 'none'}),
    html.Div(id='groups-storage', children=None, style={'display': 'none'}),
    html.Div(id='embeddings-storage', children=None, style={'display': 'none'}),

    dcc.Tabs(id="tabs", value='parameters-tab', children=[
        dcc.Tab(label='Clustering parameters', value='parameters-tab', children=parameters_layout),
        dcc.Tab(label='Results table', value='results-table-tab', children=results_table_layout),
        dcc.Tab(label='Message cluster graph', value='results-graph-tab', children=results_graph_layout)
    ]),
])


if __name__ == '__main__':
    app.run_server(debug=True)

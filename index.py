import dash_core_components as dcc
import dash_html_components as html

from os.path import join
from atexit import register
from os import remove, listdir

# the "noqa" comment tells the linter to ignore that import is unused
import webapp.callbacks  # noqa: F401
from webapp.app import app, CUSTOM_MODEL_DIR
from webapp.layouts import parameters_layout, results_table_layout, results_graph_layout, knee_graph_layout


parameters = html.Div(
    id='sidebar-parameters',
    className='col-md-3 col-lg-3 bg-light',
    children=[
        html.Div(id='parameters-layout', children=parameters_layout)
    ]
)

page_contents = html.Main(
    className='col-md-9 col-lg-9',
    children=[
        html.Div(id='results-storage', children=None, style={'display': 'none'}),
        html.Div(id='groups-storage', children=None, style={'display': 'none'}),
        html.Div(id='embeddings-storage', children=None, style={'display': 'none'}),
        html.Div(id='knee-data-storage', children=None, style={'display': 'none'}),
        html.Div(id='clustering-parameters-storage', children=None, style={'display': 'none'}),

        dcc.Tabs(value='table', style={'font-size': 'large'}, children=[
            dcc.Tab(id='table-tab', label='Cluster table', value='table', children=results_table_layout),
            dcc.Tab(id='graph-tab', label='Cluster graph', value='graph', children=results_graph_layout),
            dcc.Tab(id='knee-graph-tab', label='Knee graph', value='knee-graph', children=knee_graph_layout)
        ])
    ]
)

app.layout = html.Div(children=[
    html.Header(
        className='navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow',
        children=[
            html.A(
                className='navbar-brand col-md-3 col-lg-3 me-0 px-3',
                children="ClusterLogs"
            ),
        ]
    ),

    html.Div(className='container-fluid', children=[
        html.Div(className='row', children=[
            parameters,
            page_contents
        ])
    ])
])


def delete_temp_files() -> None:
    model_files = listdir(CUSTOM_MODEL_DIR)
    file_num = len(model_files)
    for file in model_files:
        remove(join(CUSTOM_MODEL_DIR, file))
    if file_num > 0:
        print(f"Deleted {file_num} temporary model file{'' if file_num == 1 else 's'}")


if __name__ == '__main__':
    register(delete_temp_files)
    app.run_server(debug=True)

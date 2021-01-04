import dash_core_components as dcc
import dash_html_components as html

# the "noqa" comment tells the linter to ignore that import is unused
import webapp.callbacks  # noqa: F401
from webapp.app import app
from webapp.layouts import parameters_layout, results_table_layout, results_graph_layout


navbar = html.Nav(
    id='sidebarMenu',
    className='col-md-3 col-lg-2 d-md-block bg-light sidebar collapse',
    children=[html.Div(className='position-sticky pt-3', children=[
        html.Ul(
            className='nav flex-column',
            children=[
                html.Li(
                    className='nav-item',
                    children=[
                        dcc.Link(
                            className='nav-link',
                            children="Parameters",
                            href='/'
                        )
                    ]
                ),

                html.Li(
                    className='nav-item',
                    children=[
                        dcc.Link(
                            className='nav-link',
                            children="Results",
                            href='/results-table'
                        )
                    ]
                ),

                html.Li(
                    className='nav-item',
                    children=[
                        dcc.Link(
                            className='nav-link',
                            children="Cluster Graph",
                            href='/results-graph'
                        )
                    ]
                )
            ]
        )
    ])]
)

page_contents = html.Main(
    className='col-md-9 ms-sm-auto col-lg-10 px-md-4',
    children=[
        dcc.Location(id='url', refresh=False),

        html.Div(id='results-storage', children=None, style={'display': 'none'}),
        html.Div(id='groups-storage', children=None, style={'display': 'none'}),
        html.Div(id='embeddings-storage', children=None, style={'display': 'none'}),

        html.Div(id='parameters-layout', children=parameters_layout),
        html.Div(id='results-table-layout', children=results_table_layout),
        html.Div(id='results-graph-layout', children=results_graph_layout),
    ]
)

app.layout = html.Div([
    html.Header(
        className='navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow',
        children=[
            html.A(
                className='navbar-brand col-md-3 col-lg-2 me-0 px-3',
                children="ClusterLogs"
            ),
        ]
    ),

    html.Div(className='container-fluid', children=[
        html.Div(className='row', children=[
            navbar,
            page_contents
        ])
    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)

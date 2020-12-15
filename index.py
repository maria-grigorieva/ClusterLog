import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output

import webapp.callbacks  # noqa: F401
from webapp.app import app
from webapp.layouts import parameters_layout


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return parameters_layout
    # elif pathname == '':
    #     return layout2
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True)

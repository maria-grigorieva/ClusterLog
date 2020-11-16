import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html

from collections.abc import Iterable
from dash.dependencies import Input, Output, State

from clusterlogs.pipeline import Chain


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def process(filename, model_name):
    df = pd.read_csv(filename)
    target = 'exeerrordiag'
    cluster = Chain(df, target, model_name=model_name, mode='process',
                    add_placeholder=False, matching_accuracy=0.8, output_type='html',
                    clustering_type='kmeans', keywords_extraction='lda')
    cluster.process()
    return cluster.result


def generate_table(dataframe, columns=None, max_rows=10):
    if columns is None:
        columns = dataframe.columns

    def format_item(item):
        if isinstance(item, Iterable):
            return html.Ul([
                html.Li(x) for x in item
            ])
        return item

    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(format_item(dataframe.iloc[i][col])) for col in columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app.layout = html.Div(children=[
    html.H1(children='ClusterLogs'),

    html.Div(["Enter name of csv file: ",
              dcc.Input(id='input-file', value='', type='text')]),

    html.Div(["Enter name of model file: ",
              dcc.Input(id='model-file', value='', type='text')]),

    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

    html.Div(id='results-table',
             children=None)
])


# ./samples/exeerror_1week.csv
# ./models/exeerrors_01-01-20_05-20-20.model
@app.callback(
    Output(component_id='results-table', component_property='children'),
    [Input('submit-button-state', 'n_clicks')],
    [State(component_id='input-file', component_property='value'),
     State(component_id='model-file', component_property='value')])
def update_table(_, filename, model_name):
    if not filename or not model_name:
        return None
    return generate_table(process(filename, model_name), columns=['cluster_size', 'pattern', 'common_phrases'])


if __name__ == '__main__':
    app.run_server(debug=True)

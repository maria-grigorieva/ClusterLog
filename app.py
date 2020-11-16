import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html

from collections.abc import Iterable

from clusterlogs.pipeline import Chain


def process():
    df = pd.read_csv('./samples/exeerror_1week.csv')
    target = 'exeerrordiag'
    cluster = Chain(df, target, model_name='./models/exeerrors_01-01-20_05-20-20.model', mode='process',
                    add_placeholder=False, matching_accuracy=0.8, output_type='html',
                    clustering_type='kmeans', keywords_extraction='lda')
    cluster.process()
    return cluster.result


def upload_file():
    pass


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


def main():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    df = process()

    app.layout = html.Div(children=[
        html.H1(children='ClusterLogs'),

        html.Div(children='''
            Clusterlogs results webapp, test variant
        '''),

        generate_table(df, columns=['cluster_size', 'pattern', 'common_phrases'])
    ])

    app.run_server(debug=True)


if __name__ == '__main__':
    main()

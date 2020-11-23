import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html

from collections.abc import Iterable
from dash.dependencies import Input, Output, State

from clusterlogs.pipeline import Chain


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def process(filename, target_column,
            model_name, tokenizer_type,
            clustering_algorithm, keywords_extraction):
    df = pd.read_csv(filename)
    cluster = Chain(df, target_column, model_name=model_name, mode='process',
                    add_placeholder=False, matching_accuracy=0.8, output_type='html',
                    tokenizer_type=tokenizer_type,
                    clustering_type=clustering_algorithm,
                    keywords_extraction=keywords_extraction)
    cluster.process()
    return cluster.result


def generate_table(dataframe, columns=None, max_rows=None):
    if columns is None:
        columns = dataframe.columns
    if max_rows is None or max_rows > len(dataframe):
        max_rows = len(dataframe)

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
            ]) for i in range(max_rows)
        ])
    ])


app.layout = html.Div(children=[
    html.H1(children='ClusterLogs'),

    html.H5(children='Files with data to cluster'),

    # TODO: Remove test values after app is done

    html.Br(),
    html.Div([
        html.P(
            children=[
                html.Span(children="Enter name of csv file: ",
                          style={"display": "table-cell"}),
                dcc.Input(id='input-file', value='./samples/exeerror_1week.csv', type='text',
                          style={"display": "table-cell", "width": "200%"})],
            style={"display": "table-row"}),

        html.P(
            children=[
                html.Span(children="Enter name of csv column with error description: ",
                          style={"display": "table-cell"}),
                dcc.Input(id='target-column', value='exeerrordiag', type='text',
                          style={"display": "table-cell", "width": "200%"})],
            style={"display": "table-row"}),

        html.P(
            children=[
                html.Span(children="Enter name of word2vec model file: ",
                          style={"display": "table-cell"}),
                dcc.Input(id='model-file', value='./models/exeerrors_01-01-20_05-20-20.model', type='text',
                          style={"display": "table-cell", "width": "200%"})],
            style={"display": "table-row"})],

        style={"display": "table", "border-spacing": "15px 0px", "margin": "-15px"}),

    html.Br(),
    html.H5(children='Log clustering parameters'),

    html.Div(
        ["Tokenizer type: ",
         dcc.Dropdown(
             id='tokenizer-type',
             options=[
                {'label': 'Conservative', 'value': 'conservative'},
                {'label': 'Aggressive', 'value': 'aggressive'},
                {'label': 'Space', 'value': 'space'},
             ],
             value='space'),

         "Clustering algorithm: ",
         dcc.Dropdown(
             id='clustering-algorithm',
             options=[
                {'label': 'Similarity', 'value': 'similarity'},
                {'label': 'K-means', 'value': 'kmeans'},
                {'label': 'DBSCAN', 'value': 'dbscan'},
                {'label': 'OPTICS', 'value': 'optics'},
                {'label': 'HDBSCAN', 'value': 'hdbscan'},
                {'label': 'Hierarchical', 'value': 'hierarchical'},
             ],
             value='dbscan'),

         "Keyword extraction algorithm: ",
         dcc.Dropdown(
             id='keyword-extraction-algorithm',
             options=[
                {'label': 'RAKE', 'value': 'RAKE'},
                {'label': 'RAKE (nltk version)', 'value': 'rake_nltk'},
                {'label': 'Latent Dirichle Allocation', 'value': 'lda'},
                {'label': 'N-grams', 'value': 'ngrams'},
                {'label': 'PageRank (Gensim)', 'value': 'gensim'},
                {'label': 'TF-IDF', 'value': 'tfidf'},
                {'label': 'KPMiner', 'value': 'KPMiner'},
                {'label': 'YAKE', 'value': 'YAKE'},
                {'label': 'TextRank (pke)', 'value': 'TextRank'},
                {'label': 'TextRank (pyTextRank)', 'value': 'pyTextRank'},
                {'label': 'SingleRank', 'value': 'SingleRank'},
                {'label': 'TopicRank', 'value': 'TopicRank'},
                {'label': 'TopicalPageRank', 'value': 'TopicalPageRank'},
                {'label': 'PositionRank', 'value': 'PositionRank'},
                {'label': 'MultipartiteRank', 'value': 'MultipartiteRank'},
                {'label': 'Kea', 'value': 'Kea'},
                {'label': 'WINGNUS', 'value': 'WINGNUS'},
             ],
             value='rake_nltk')]),

    html.Div(
        dcc.Checklist(
            id='boolean-options',
            options=[
                {'label': ' Add placeholders', 'value': 'add_placeholder'},
                {'label': ' Dimensionality reduction', 'value': 'dimensionality_reduction'},
                {'label': ' Perform categorization', 'value': 'categorization'}
            ],
            value=[]
        ),
    ),

    html.Br(),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

    html.Hr(),
    html.Div(id='results-table',
             children=None)
])

# Chain parameters: implemented are marked with +
# + df,
# + target,
# + tokenizer_type='space',
# cluster_settings=None,
# + model_name='word2vec.model',
# mode='create',
# output_type='csv',
# output_fname='report',
# + add_placeholder=True,
# + dimensionality_reduction=False,
# threshold=5000,
# matching_accuracy=0.8,
# + clustering_type='dbscan',
# + keywords_extraction='rake_nltk',
# + categorization=False

# Test values
# ./samples/exeerror_1week.csv
# exeerrordiag
# ./models/exeerrors_01-01-20_05-20-20.model


@app.callback(
    Output(component_id='results-table', component_property='children'),
    [Input('submit-button-state', 'n_clicks')],
    [State(component_id='input-file', component_property='value'),
     State(component_id='target-column', component_property='value'),
     State(component_id='model-file', component_property='value'),
     State(component_id='tokenizer-type', component_property='value'),
     State(component_id='clustering-algorithm', component_property='value'),
     State(component_id='keyword-extraction-algorithm', component_property='value'),
     State(component_id='boolean-options', component_property='value')])
def update_table(n_clicks, filename,
                 target_column, model_name,
                 tokenizer_type, clustering_algorithm,
                 keywords_extraction, boolean_options):

    if n_clicks == 0 or not filename or not target_column:
        return None

    options = {
        'add_placeholder': False,
        'dimensionality_reduction': False,
        'categorization': False
    }
    for option in boolean_options:
        options[option] = True

    return generate_table(process(filename,
                                  target_column,
                                  model_name,
                                  tokenizer_type,
                                  clustering_algorithm,
                                  keywords_extraction),
                          columns=['cluster_size', 'pattern', 'common_phrases'])


if __name__ == '__main__':
    app.run_server(debug=True)

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html

from io import StringIO
from os.path import exists
from base64 import b64decode
from collections.abc import Iterable
from typing import List, Optional, Dict
from dash.dependencies import Input, Output, State

from clusterlogs.pipeline import Chain


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def process(dataframe: pd.DataFrame, target_column: str,
            model_name: str, update_model: bool, tokenizer_type: str,
            clustering_algorithm: str, keywords_extraction: str,
            options: Dict[str, bool], threshold: int, matching_accuracy: float) -> pd.DataFrame:
    mode = 'process'
    if update_model:
        mode = 'update' if exists(model_name) else 'create'
    cluster = Chain(dataframe, target_column,
                    model_name=model_name, mode=mode,
                    add_placeholder=options['add_placeholder'],
                    dimensionality_reduction=options['dimensionality_reduction'],
                    categorization=options['categorization'],
                    threshold=threshold,
                    matching_accuracy=matching_accuracy,
                    output_type='html',
                    tokenizer_type=tokenizer_type,
                    clustering_type=clustering_algorithm,
                    keywords_extraction=keywords_extraction)
    cluster.process()
    return cluster.result


def generate_table(dataframe: pd.DataFrame, columns: Optional[List[str]] = None, max_rows: Optional[int] = None) -> html.Table:
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

    html.Div(
        dcc.Upload(
            id='input-file',
            children=html.Div([
                'Drag and drop or ',
                html.A('select a csv file')
            ]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
        )
    ),

    html.Br(),
    html.Div([
        # html.P(
        #     children=[
        #         html.Span(children="Enter name of csv file: ",
        #                   style={"display": "table-cell"}),
        #         dcc.Input(id='input-file', value='./samples/exeerror_1week.csv', type='text',
        #                   style={"display": "table-cell", "width": "200%"})],
        #     style={"display": "table-row"}),

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

    html.Div(
        dcc.Checklist(
            id='update-model',
            options=[
                {'label': ' Use input data in word2vec model', 'value': 'update_model'},
            ],
            value=[]
        ),
    ),

    html.Br(),

    dcc.ConfirmDialog(
        id='no-model-warning',
        message='Word2vec model file does not exist.\nEnter a valid path or use input data to create a new model',
        displayed=False
    ),

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

    html.Br(),

    html.Div([
        html.P(
            children=[
                html.Span(children="Similarity threshold",
                          style={"display": "table-cell"}),
                dcc.Input(id='threshold', value=5000, type='number', debounce=True, min=1,
                          style={"display": "table-cell"})],
            style={"display": "table-row"}),

        html.P(
            children=[
                html.Span(children="Sequence matching accuracy",
                          style={"display": "table-cell"}),
                dcc.Input(id='matching-accuracy', value=0.8, type='number', debounce=True, min=0.0, max=1.0, step=0.01,
                          style={"display": "table-cell"})],
            style={"display": "table-row"})],

        style={"display": "table", "border-spacing": "15px 0px", "margin": "-15px"}),

    html.Br(),

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
# + mode='create',
# output_type='csv',
# output_fname='report',
# + add_placeholder=True,
# + dimensionality_reduction=False,
# + threshold=5000,
# + matching_accuracy=0.8,
# + clustering_type='dbscan',
# + keywords_extraction='rake_nltk',
# + categorization=False

# Test values
# ./samples/exeerror_1week.csv
# exeerrordiag
# ./models/exeerrors_01-01-20_05-20-20.model


def parse_input_file(content: str) -> pd.DataFrame:
    _, content_string = content.split(',')
    decoded = b64decode(content_string)
    df = pd.read_csv(StringIO(decoded.decode('utf-8')))
    return df


@app.callback(
    Output(component_id='input-file', component_property='children'),
    [Input(component_id='input-file', component_property='contents')],
    [State(component_id='input-file', component_property='filename')]
)
def display_uploaded_filename(contents: str, filename: str) -> html.Div:
    if contents:
        return html.Div(filename)
    else:
        return html.Div([
            'Drag and drop or ',
            html.A('select a csv file')
        ]),


@app.callback(
    Output(component_id='no-model-warning', component_property='displayed'),
    [Input('submit-button-state', 'n_clicks')],
    [State(component_id='model-file', component_property='value'),
     State(component_id='update-model', component_property='value')])
def display_model_file_warning(_, model_name: str, update_model: bool) -> bool:
    if not update_model and not exists(model_name):
        return True
    return False


@app.callback(
    Output(component_id='results-table', component_property='children'),
    [Input('submit-button-state', 'n_clicks')],
    [State(component_id='input-file', component_property='contents'),
     State(component_id='target-column', component_property='value'),
     State(component_id='model-file', component_property='value'),
     State(component_id='update-model', component_property='value'),
     State(component_id='tokenizer-type', component_property='value'),
     State(component_id='clustering-algorithm', component_property='value'),
     State(component_id='keyword-extraction-algorithm', component_property='value'),
     State(component_id='threshold', component_property='value'),
     State(component_id='matching-accuracy', component_property='value'),
     State(component_id='boolean-options', component_property='value')])
def update_table(n_clicks: int,
                 input_file: Optional[str], target_column: str,
                 model_name: str, update_model: List[str],
                 tokenizer_type: str, clustering_algorithm: str,
                 keywords_extraction: str, threshold: int,
                 matching_accuracy: float, boolean_options: List[str]) -> Optional[html.Table]:

    if n_clicks == 0 or not input_file or not target_column:
        return None
    if not update_model and not exists(model_name):
        return None

    dataframe = parse_input_file(input_file)

    options = {
        'add_placeholder': False,
        'dimensionality_reduction': False,
        'categorization': False
    }
    for option in boolean_options:
        options[option] = True

    return generate_table(process(dataframe,
                                  target_column,
                                  model_name,
                                  bool(update_model),
                                  tokenizer_type,
                                  clustering_algorithm,
                                  keywords_extraction,
                                  options,
                                  threshold,
                                  matching_accuracy),
                          columns=['cluster_size', 'pattern', 'common_phrases'])


if __name__ == '__main__':
    app.run_server(debug=True)

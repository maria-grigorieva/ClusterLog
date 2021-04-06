import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from os import listdir
from dash_extensions import Download


parameters_layout = html.Main(className='container px-6', children=[
    html.Br(),
    html.H4(children='Input data'),

    # TODO: Remove test values after app is done

    dbc.Row([
        dbc.Col(dcc.Upload(
            id='input-file',
            children="Select a file",
            className='btn btn-outline-secondary'
        ), width='auto'),
        dbc.Col(html.P(id='input-file-name', children='No csv file selected', className='font-weight-light mt-2'))
    ], className='mt-3 mb-2'),

    dbc.FormGroup([
        dbc.Label(children="Target column:", html_for='target-column'),
        dbc.Input(id='target-column', value='exeerrordiag', type='text')
    ]),

    dbc.FormGroup([
        dbc.Label(children="Transformer model:", html_for='model-file'),
        dbc.Select(
            id='model-file',
            options=[
                {'label': model_name, 'value': './models/' + model_name} for model_name in listdir('./models')
            ] + [{'label': 'Custom model', 'value': 'custom'}],
            value=None
        )
    ]),

    # dbc.FormGroup(id='custom-model', style={'display': 'none'}, children=[
    #     dbc.Label(children="Enter path to word2vec model file:", html_for='custom-model-file'),
    #     dbc.Input(id='custom-model-file', value='', type='text')
    # ]),

    dbc.Row(id='custom-model-group', style={'display': 'none'}, children=[
        dbc.Col(dcc.Upload(
            id='custom-model-upload',
            children="Select a file",
            className='btn btn-outline-secondary'
        ), width='auto'),
        dbc.Col(html.P(id='custom-model-name', children='No model file selected', className='font-weight-light mt-2'))
    ]),

    Download(id='custom-model-download'),

    dbc.FormGroup([
        dbc.Label("Transformer model usage mode", html_for='model-usage-mode'),
        dbc.RadioItems(
            options=[
                {"label": "Use an existing model", "value": 'process'},
                {"label": "Update an existing model", "value": 'update'},
                {"label": "Create a new model", "value": 'create'},
            ],
            value='process',
            id="model-usage-mode",
        ),
    ]),

    html.Hr(),

    html.H4(children='Pipeline parameters'),

    dbc.FormGroup([
        dbc.Label(children="Clustering algorithm", html_for='clustering-algorithm'),
        dbc.Select(
            id='clustering-algorithm',
            options=[
                {'label': 'Similarity', 'value': 'similarity'},
                {'label': 'K-means', 'value': 'kmeans'},
                {'label': 'DBSCAN', 'value': 'dbscan'},
                {'label': 'OPTICS', 'value': 'optics'},
                {'label': 'HDBSCAN', 'value': 'hdbscan'},
                {'label': 'Hierarchical', 'value': 'hierarchical'},
            ],
            value='dbscan'
        )
    ]),

    dbc.FormGroup([
        dbc.Label(children="Keyword extraction algorithm", html_for='keyword-extraction-algorithm'),
        dbc.Select(
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
            value='rake_nltk'
        )
    ]),

    dbc.FormGroup([
        dbc.Label(children="Vectorization type", html_for='vectorization-type'),
        dbc.Select(
            id='vectorization-type',
            options=[
                {'label': 'Word2vec', 'value': 'word2vec'},
                {'label': 'BERT', 'value': 'bert'},
            ],
            value='word2vec'
        )
    ]),

    dbc.FormGroup([
        dbc.Label(children="Tokenizer type", html_for='tokenizer-type'),
        dbc.Select(
            id='tokenizer-type',
            options=[
                {'label': 'Conservative', 'value': 'conservative'},
                {'label': 'Aggressive', 'value': 'aggressive'},
                {'label': 'Space', 'value': 'space'},
            ],
            value='space'
        )
    ]),

    dbc.FormGroup([
        dbc.Label(children="Sequence matching accuracy", html_for='matching-accuracy'),
        dbc.Input(id='matching-accuracy', value=0.8, type='number', debounce=True, min=0.0, max=1.0, step=0.01)
    ]),

    dbc.Checklist(
        id='boolean-options',
        options=[
            {'label': ' Add placeholders', 'value': 'add_placeholder'},
            {'label': ' Dimensionality reduction', 'value': 'dimensionality_reduction'},
        ],
        value=['add_placeholder'],
    ),

    html.Div(id='clustering-parameters-div', children=[
        html.Hr(),
        html.H4("Clustering method parameters"),

        dbc.FormGroup(id='form-metric', children=[
            dbc.Label(children="Metric", html_for='params-metric'),
            dbc.Select(
                id='params-metric',
                options=[
                    {'label': 'Euclidean', 'value': 'euclidean'},
                    {'label': 'Manhattan', 'value': 'manhattan'},
                    {'label': 'Cosine', 'value': 'cosine'},
                ],
                value='euclidean'
            )
        ]),
        dbc.FormGroup(id='form-epsilon', children=[
            dbc.Label(children="Maximum neighbour distance (epsilon)", html_for='params-epsilon'),
            dbc.Input(id='params-epsilon', value=None, placeholder="Leave empty to choose automatically", type='number', step=0.01, debounce=True)
        ]),
        dbc.FormGroup(id='form-min-samples', children=[
            dbc.Label(children="Number of neighbours for core point", html_for='params-min-samples'),
            dbc.Input(id='params-min-samples', value=1, type='number', debounce=True, min=1, max=1000, step=1)
        ]),
        dbc.FormGroup(id='form-cluster-number', children=[
            dbc.Label(children="Cluster number", html_for='params-cluster-number'),
            dbc.Input(id='params-cluster-number', value=30, type='number', debounce=True, min=1, max=1000, step=1)
        ]),
    ]),

    html.Hr(),
    html.Div(id='word2vec-parameters-group', children=[
        html.H4("Word2Vec parameters"),
        dbc.FormGroup([
            dbc.Label(children="Word2Vec vector size", html_for='w2v-vector-size'),
            dbc.Input(id='w2v-vector-size', value=300, type='number', debounce=True, min=1, max=1000, step=1)
        ]),
        dbc.FormGroup([
            dbc.Label(children="Word2Vec window width", html_for='w2v-window'),
            dbc.Input(id='w2v-window', value=7, type='number', debounce=True, min=1, max=20, step=1)
        ]),
    ]),

    html.Br(),
    dbc.Button(id='submit-button-state', n_clicks=0, className='mr-1', color='primary',
               children=[dbc.Spinner(children=[html.Div(id='loading-output')], size='sm')]),
    dbc.Row()
])

results_graph_layout = html.Div(className='container px-6', children=[
    html.Br(),
    dbc.Label(children="Noise threshold", html_for='noise-threshold'),
    dcc.Slider(
        id='noise-threshold',
        min=0,
        max=1000,
        step=1,
        value=100,
        tooltip={'always_visible': True, 'placement': 'top'}
    ),
    html.Div(id='results-graph', children=None),
])

results_table_layout = html.Div(className='container px-6', children=[
    html.Br(),
    html.Div(id='results-table', children=None),
])

knee_graph_layout = html.Div(className='container px-6', children=[
    html.Div(id='knee-graph', children=None),
])

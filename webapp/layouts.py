import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from os import listdir


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
        dbc.Col(html.P(id='input-file-name', children='No csv file selected', className='lh-lg fs-6 fw-light'))
    ]),

    dbc.Row(
        dbc.Col(
            dbc.FormGroup([
                dbc.Label(children="Target column:", html_for='target-column'),
                dbc.Input(id='target-column', value='exeerrordiag', type='text')
            ]),
            width=6
        )
    ),

    dbc.Row(children=[
        dbc.Col(
            dbc.FormGroup([
                dbc.Label(children="Word2vec model:", html_for='model-file'),
                dbc.Select(
                    id='model-file',
                    options=[
                        {'label': model_name, 'value': './models/' + model_name} for model_name in listdir('./models')
                    ] + [{'label': 'Custom model', 'value': 'custom'}],
                    value=None
                )
            ]),
            width=6
        ),
        dbc.Col(
            dbc.FormGroup([
                dbc.Label(children="Enter path to word2vec model file:", html_for='custom-model-file'),
                dbc.Input(id='custom-model-file', value='', type='text')
            ]),
            width=6,
            style={'display': 'none'},
            id='custom-model'
        )
    ]),

    dbc.Checklist(
        id='update-model',
        options=[
            {'label': ' Use input data in word2vec model', 'value': 'update_model'},
        ],
        value=[]
    ),

    html.Hr(),

    dcc.ConfirmDialog(
        id='no-model-warning',
        message='Word2vec model file does not exist.\nEnter a valid path or use input data to create a new model',
        displayed=False
    ),

    dbc.Row(
        dbc.Col([
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
            ])],
            width=6
        ),
    ),

    dbc.Row(
        dbc.Col([
            dbc.FormGroup([
                dbc.Label(children="Sequence matching accuracy", html_for='matching-accuracy'),
                dbc.Input(id='matching-accuracy', value=0.8, type='number', debounce=True, min=0.0, max=1.0, step=0.01)
            ])],
            width=4
        )
    ),

    dbc.Checklist(
        id='boolean-options',
        options=[
            {'label': ' Add placeholders', 'value': 'add_placeholder'},
            {'label': ' Dimensionality reduction', 'value': 'dimensionality_reduction'},
        ],
        value=[],
    ),

    html.Br(),
    dbc.Button(id='submit-button-state', n_clicks=0, children=['Submit'], className='mr-1', color='primary'),
    dbc.Row()
])

results_layout = html.Main(className='container px-6', children=[
    html.Br(),
    html.H4("Log message clusters"),
    html.Div(id='results-graph', children=None),
    html.Hr(),
    html.H4("Cluster table"),
    html.Div(id='results-table', children=None),
])

knee_graph_layout = html.Main(className='container px-6', children=[
    html.Br(),
    html.H4("Knee points"),
    html.Div(id='knee-graph', children=None),
])

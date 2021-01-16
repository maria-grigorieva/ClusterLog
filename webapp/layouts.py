import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


parameters_layout = html.Main(className='container px-6', children=[
    html.Br(),
    html.H4(children='Files with data to cluster'),

    # TODO: Remove test values after app is done

    dbc.Row([
        dbc.Col(dcc.Upload(
            id='input-file',
            children="Select a log file",
            className='btn btn-outline-secondary'
        ), width='auto'),
        dbc.Col(html.P(id='input-file-name', children='No csv file selected', className='lh-lg fs-6 fw-light'))
    ]),

    dbc.Row(
        dbc.Col(
            dbc.FormGroup([
                dbc.Label(children="Enter name of csv column with error description:", html_for='target-column'),
                dbc.Input(id='target-column', value='exeerrordiag', type='text')
            ]),
            width=6
        )
    ),

    dbc.Row(
        dbc.Col(
            dbc.FormGroup([
                dbc.Label(children="Enter name of word2vec model file:", html_for='model-file'),
                dbc.Input(id='model-file', value='./models/exeerrors_01-01-20_05-20-20.model', type='text')
            ]),
            width=6
        )
    ),

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

    html.H4(children='Log clustering parameters'),

    html.Div(
        [html.Label(children="Tokenizer type", className='form-label'),
         #  "Tokenizer type: ",
         dcc.Dropdown(
             id='tokenizer-type',
             className='col-lg-7',
             options=[
                {'label': 'Conservative', 'value': 'conservative'},
                {'label': 'Aggressive', 'value': 'aggressive'},
                {'label': 'Space', 'value': 'space'},
             ],
             value='space'),

         "Clustering algorithm: ",
         dcc.Dropdown(
             id='clustering-algorithm',
             className='col-lg-7',
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
             className='col-lg-7',
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
                          style={"display": "table-cell"}, className='input-group')],
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
            value=[],
            className='form-check'
        ),
    ),

    html.Br(),
    dbc.Button(id='submit-button-state', n_clicks=0, children='Submit', className='mr-1', color='secondary'),
])

results_table_layout = html.Div(children=[
    html.Div(id='results-table', className='table-responsive', children=None),
])

results_graph_layout = html.Div(children=[
    html.Div(id='results-graph', className='my-4 w-100', children=None),
])

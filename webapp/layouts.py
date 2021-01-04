import dash_core_components as dcc
import dash_html_components as html


parameters_layout = html.Div(children=[
    html.Br(),
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
])

results_table_layout = html.Div(children=[
    html.Div(id='results-table', children=None),
])

results_graph_layout = html.Div(children=[
    html.Div(id='results-graph', children=None),
])

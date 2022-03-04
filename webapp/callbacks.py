import numpy as np
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html

from math import log
from json import dumps, loads
from ntpath import basename
from base64 import b64encode
from os import listdir
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple, Dict, Any, Union

from sklearn.manifold import TSNE
from plotly.graph_objs import Figure, Scatter
from dash.dependencies import Input, Output, State

from webapp.process import execute_pipeline
from webapp.app import app, CUSTOM_MODEL_DIR
from webapp.utility import parse_input_file, parse_model_file, generate_table


def send_file(path, filename=None, mime_type=None):
    """
    Convert a file into the format expected by the Download component.
    :param path: path to the file to be sent
    :param filename: name of the file, if not provided the original filename is used
    :param mime_type: mime type of the file (optional, passed to Blob in the javascript layer)
    :return: dict of file content (base64 encoded) and meta data used by the Download component
    """
    # If filename is not set, read it from the path.
    if filename is None:
        filename = basename(path)
    # Read the file into a base64 string.
    with open(path, "rb") as f:
        content = b64encode(f.read()).decode()
    # Wrap in dict.
    return dict(content=content, filename=filename, mime_type=mime_type, base64=True)


@app.callback(
    Output(component_id='input-file-name', component_property='children'),
    [Input(component_id='input-file', component_property='contents')],
    [State(component_id='input-file', component_property='filename')]
)
def display_uploaded_filename(contents: str, filename: str) -> str:
    if contents:
        return html.Div(filename)
    else:
        return "No csv file selected"


@app.callback(
    Output(component_id='custom-model-name', component_property='children'),
    [Input(component_id='custom-model-upload', component_property='contents')],
    [State(component_id='custom-model-upload', component_property='filename')]
)
def display_custom_model_filename(contents: str, filename: str) -> str:
    if contents:
        return html.Div(filename)
    else:
        return "No model file selected"

# @app.callback(
#     Output(component_id='no-model-warning', component_property='displayed'),
#     [Input(component_id='submit-button-state', component_property='n_clicks')],
#     [State(component_id='model-file', component_property='value'),
#      State(component_id='custom-model-file', component_property='value'),
#      State(component_id='model-usage-mode', component_property='value')])
# def display_model_file_warning(n_clicks: int, model_file: str, custom_model_file: str, model_usage_mode: str) -> bool:
#     if n_clicks == 0:
#         return False
#     if model_usage_mode != 'create' and model_file == 'custom' and not exists(custom_model_file):
#         return True
#     return False


@app.callback(
    Output(component_id='results-storage', component_property='children'),
    Output(component_id='groups-storage', component_property='children'),
    Output(component_id='embeddings-storage', component_property='children'),
    Output(component_id='knee-data-storage', component_property='children'),
    Output(component_id='loading-output', component_property='children'),
    Output(component_id='custom-model-download', component_property='data'),
    [Input(component_id='submit-button-state', component_property='n_clicks')],
    [State(component_id='input-file', component_property='contents'),
     State(component_id='target-column', component_property='value'),
     State(component_id='model-file', component_property='value'),
     State(component_id='custom-model-upload', component_property='contents'),
     State(component_id='model-usage-mode', component_property='value'),
     State(component_id='tokenizer-type', component_property='value'),
     State(component_id='clustering-algorithm', component_property='value'),
     State(component_id='clustering-parameters-storage', component_property='children'),
     State(component_id='keyword-extraction-algorithm', component_property='value'),
     State(component_id='matching-accuracy', component_property='value'),
     State(component_id='boolean-options', component_property='value'),
     State(component_id='w2v-vector-size', component_property='value'),
     State(component_id='w2v-window', component_property='value')])
def update_results(
        n_clicks: int,
        input_file: Optional[str], target_column: str,
        model_name: str, custom_model: Optional[str], model_usage_mode: str,
        tokenizer_type: str, clustering_algorithm: str,
        clustering_parameters_json: str,
        keywords_extraction: str, matching_accuracy: float,
        boolean_options: List[str], word2vec_size: int,
        word2vec_window: int) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str, Optional[Dict[Any, Union[str, bool]]]]:

    if n_clicks == 0 or not input_file or not target_column:
        return None, None, None, None, "Submit", None

    dataframe = parse_input_file(input_file)

    custom_model_name = ''
    if model_name == 'custom':
        if custom_model is None:
            if model_usage_mode != 'create':
                return None, None, None, None, "Submit", None
            else:
                new_model_file = NamedTemporaryFile(mode='w+b', dir=CUSTOM_MODEL_DIR, delete=False, suffix='.model')
                model_name = custom_model_name = new_model_file.name
        else:
            model_name = custom_model_name = parse_model_file(custom_model)

    options = {
        'add_placeholder': False,
        'dimensionality_reduction': False,
    }
    for option in boolean_options:
        options[option] = True

    word2vec_parameters = {
        'w2v_size': word2vec_size,
        'w2v_window': word2vec_window
    }

    clustering_parameters = loads(clustering_parameters_json)

    result = execute_pipeline(dataframe, target_column,
                              model_name, model_usage_mode,
                              tokenizer_type, clustering_algorithm,
                              clustering_parameters,
                              keywords_extraction, options,
                              matching_accuracy, word2vec_parameters)

    groups: pd.DataFrame = result.groups

    knee_data = None
    if clustering_algorithm == 'dbscan':
        knee_data = dumps(result.clusters.knee_data)

    if clustering_algorithm != "similarity":
        embeddings: np.ndarray = result.vectors.sent2vec
        jsoned_embeddings = pd.DataFrame(embeddings).to_json(orient='split')
    else:
        jsoned_embeddings = None

    download_file = None
    if custom_model_name and model_usage_mode != 'process':
        download_file = send_file(custom_model_name)

    return result.result.to_json(), groups.to_json(), jsoned_embeddings, knee_data, "Submit", download_file


@app.callback(
    Output(component_id='results-table', component_property='children'),
    [Input('results-storage', 'children')])
def update_table(stored_results: str) -> Optional[html.Table]:
    if not stored_results:
        return None
    result: pd.DataFrame = pd.read_json(stored_results)  # type: ignore
    return generate_table(result, columns=['cluster_size', 'patterns', 'common_phrases'])


@app.callback(
    Output(component_id='results-graph', component_property='children'),
    [Input('groups-storage', 'children'),
     Input('embeddings-storage', 'children'),
     Input('noise-threshold', 'value')])
def update_graph(stored_groups: str, stored_embeddings: str, noise_threshold: int) -> Optional[dcc.Graph]:
    if not stored_embeddings or not stored_groups:
        return None

    groups: pd.DataFrame = pd.read_json(stored_groups)  # type: ignore
    embeddings = pd.read_json(stored_embeddings, orient='split').to_numpy()

    tsne = TSNE(perplexity=25, random_state=42, verbose=False)
    embeddings = tsne.fit_transform(embeddings)

    @dataclass
    class Cluster():
        x: List[float]
        y: List[float]
        hover_text: List[str]
        size: List[int]
        marker_size: List[float]

        def __add__(self, other):
            if not isinstance(other, Cluster):
                return NotImplemented
            return Cluster(
                x=self.x + other.x,
                y=self.y + other.y,
                hover_text=self.hover_text + other.hover_text,
                size=self.size + other.size,
                marker_size=self.marker_size + other.marker_size,
            )

    clusters: Dict[int, Cluster] = {}
    for i, row in groups.iterrows():
        label: int = row['cluster']
        pattern = row['pattern'].replace("; ", ";<br>")

        cluster = Cluster(
            x=[embeddings[i, 0]],
            y=[embeddings[i, 1]],
            hover_text=[pattern],
            size=[row['cluster_size']],
            marker_size=[log(row['cluster_size'])]
        )

        if label not in clusters:
            clusters[label] = cluster
        else:
            clusters[label] += cluster

    noise = clusters.pop(-1, Cluster([], [], [], [], []))
    clusters = {i + 1: cluster for i, cluster in enumerate(sorted(clusters.values(), reverse=True, key=lambda c: sum(c.size)))}

    max_size = max([max(cluster.marker_size) for cluster in clusters.values()] + noise.marker_size)
    noise.marker_size = [max(7, size * 25 / max_size) for size in noise.marker_size]

    for label, cluster in clusters.copy().items():
        cluster.marker_size = [max(7, size * 25 / max_size) for size in cluster.marker_size]

        if sum(cluster.size) <= noise_threshold:
            noise += cluster
            del clusters[label]
            continue

        hover_text = []
        for size, pattern in zip(cluster.size, cluster.hover_text):
            hover_text.append(f"{size} message{'s' if size > 1 else ''} in cluster №{label}:<br>{pattern}")
        cluster.hover_text = hover_text

    noise_hover_text = []
    for size, pattern in zip(noise.size, noise.hover_text):
        noise_hover_text.append(f"{size} message{'s' if size > 1 else ''} in noise cluster:<br>{pattern}")
    noise.hover_text = noise_hover_text

    fig = Figure()
    for label, cluster in clusters.items():
        fig.add_trace(Scatter(
            x=cluster.x, y=cluster.y,
            name="Cluster № " + str(label),
            marker=dict(
                size=cluster.marker_size,
                opacity=0.8,
                line=dict(
                    width=1
                ),
            ),
            # marker_line_width=1,
            # marker_size=cluster.marker_size,
            # opacity=.8,
            text=cluster.hover_text,
        ))

    fig.add_trace(Scatter(
        x=noise.x, y=noise.y,
        name="Noise",
        marker=dict(
            size=cluster.marker_size,
            opacity=0.8,
            color='gray',
            line=dict(
                width=1
            ),
        ),
        # marker_line_width=1,
        # marker_size=noise.marker_size,
        #marker_color='gray',
        #opacity=.8,
        text=noise.hover_text,
    ))

    fig.update_traces(mode="markers", hoverinfo="text")
    fig.update_layout(hovermode="closest",
                      xaxis={"visible": False},
                      yaxis={"visible": False},
                      showlegend=True)

    return dcc.Graph(figure=fig, responsive=True, style={'height': '90vh'})


@app.callback(
    Output(component_id='knee-graph', component_property='children'),
    [Input('knee-data-storage', 'children')])
def update_knee_graph(knee_data_json: str) -> Optional[dcc.Graph]:
    if not knee_data_json:
        return None

    knee_data = loads(knee_data_json)

    fig = Figure()
    fig.add_scatter(
        x=knee_data['x'],
        y=knee_data['y'],
        name="Data",
        mode='lines',
        line=dict(
            color='blue',
        ),
        # line_color="blue"
    )

    for knee in knee_data['knees']:
        fig.add_vline(
            x=knee,
            line=dict(
                color='black',
                dash='dash',
            ),
            # line_color='black',
            # line_dash='dash'
        )

    fig.add_vline(
        x=knee_data['chosen_knee'],
        annotation_text=" Chosen epsilon value",
        annotation_font_size=15,
        line=dict(
            color='red',
            dash='dash',
        ),
        # line_color='red',
        # line_dash='dash'
    )

    fig.update_yaxes(title_text='Number of points', title_font={"size": 18}, tickfont={'size': 14})
    fig.update_xaxes(title_text='Average k-neighbours distance', title_font={"size": 18}, tickfont={'size': 14})

    return dcc.Graph(figure=fig, responsive=True, style={'height': '90vh'})


@app.callback(
    [Output('graph-tab', 'disabled'),
     Output('knee-graph-tab', 'disabled')],
    [Input('clustering-algorithm', 'value')]
)
def disable_tabs(clustering_algorithm: str) -> Tuple[bool, bool]:
    disable_graph, disable_knee_graph = False, True
    if clustering_algorithm == 'similarity':
        disable_graph = True
    if clustering_algorithm == 'dbscan':
        disable_knee_graph = False
    return disable_graph, disable_knee_graph


@app.callback(
    [Output(component_id='model-file', component_property='options'),
     Output(component_id='model-file', component_property='value')],
    [Input(component_id='model-usage-mode', component_property='value')]
)
def immutable_models_on_server(usage_mode: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    model_names = [{
        'label': model_name,
        'value': './models/' + model_name,
        'disabled': False if usage_mode == 'process' else True
    } for model_name in listdir('./models')]
    model_file = None if usage_mode == 'process' else 'custom'
    return model_names + [{'label': 'Custom model', 'value': 'custom'}], model_file


@app.callback(
    Output('custom-model-group', 'style'),
    [Input('model-file', 'value'),
     Input('model-usage-mode', 'value')]
)
def custom_model_form_visibility(model_name: str, usage_mode: str) -> Optional[Dict[str, str]]:
    if model_name == 'custom' and usage_mode != 'create':
        return None
    return {'display': 'none'}


@app.callback(
    Output('word2vec-parameters-group', 'style'),
    [Input('clustering-algorithm', 'value')]
)
def word2vec_parameters_visibility(clustering_method: str) -> Optional[Dict[str, str]]:
    if clustering_method == 'similarity':
        return {'display': 'none'}
    return None


@app.callback(
    [Output('clustering-parameters-div', 'style'),
     Output('form-metric', 'style'),
     Output('form-epsilon', 'style'),
     Output('form-min-samples', 'style'),
     Output('form-cluster-number', 'style')],
    [Input('clustering-algorithm', 'value')]
)
def parameter_visibility(clustering_method: str) -> Tuple[Optional[Dict[str, str]], ...]:
    div_style = {'display': 'none'} if clustering_method == 'similarity' else None
    parameter_styles: Dict[str, Optional[Dict[str, str]]] = {
        'metric': {'display': 'none'},
        'epsilon': {'display': 'none'},
        'min_samples': {'display': 'none'},
        'cluster_number': {'display': 'none'}
    }
    algorithm_parameters: Dict[str, List[str]] = {
        'similarity': [],
        'kmeans': ['cluster_number'],
        'dbscan': ['metric', 'epsilon', 'min_samples'],
        'optics': ['metric', 'min_samples'],
        'hdbscan': ['metric', 'min_samples'],
        'hierarchical': ['metric', 'epsilon']
    }
    for parameter in algorithm_parameters[clustering_method]:
        parameter_styles[parameter] = None
    return (div_style, ) + tuple(parameter_styles.values())


@app.callback(
    Output('clustering-parameters-storage', 'children'),
    [Input('params-metric', 'value'),
     Input('params-epsilon', 'value'),
     Input('params-min-samples', 'value'),
     Input('params-cluster-number', 'value')]
)
def get_clustering_parameters(metric: str, epsilon: Optional[float], min_samples: int, cluster_number: int) -> Optional[str]:
    parameters = {
        'metric': metric,
        'epsilon': epsilon,
        'min_samples': min_samples,
        'cluster_number': cluster_number
    }
    return dumps(parameters)

import numpy as np
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html

from math import log
from os.path import exists
from json import dumps, loads
from dataclasses import dataclass
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple, Dict
from plotly.graph_objects import Figure, Scatter
from dash.dependencies import Input, Output, State

from webapp.app import app
from webapp.process import execute_pipeline
from webapp.utility import parse_input_file, generate_table


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
    Output(component_id='no-model-warning', component_property='displayed'),
    [Input('submit-button-state', 'n_clicks')],
    [State(component_id='model-file', component_property='value'),
     State(component_id='custom-model-file', component_property='value'),
     State(component_id='update-model', component_property='value')])
def display_model_file_warning(n_clicks, model_file: str, custom_model_file: str, update_model: bool) -> bool:
    if n_clicks == 0:
        return False
    if not update_model and model_file == 'custom' and not exists(custom_model_file):
        return True
    return False


@app.callback(
    Output(component_id='results-storage', component_property='children'),
    Output(component_id='groups-storage', component_property='children'),
    Output(component_id='embeddings-storage', component_property='children'),
    Output(component_id='knee-data-storage', component_property='children'),
    Output(component_id='loading-output', component_property='children'),
    [Input(component_id='submit-button-state', component_property='n_clicks')],
    [State(component_id='input-file', component_property='contents'),
     State(component_id='target-column', component_property='value'),
     State(component_id='model-file', component_property='value'),
     State(component_id='custom-model-file', component_property='value'),
     State(component_id='update-model', component_property='value'),
     State(component_id='tokenizer-type', component_property='value'),
     State(component_id='clustering-algorithm', component_property='value'),
     State(component_id='keyword-extraction-algorithm', component_property='value'),
     State(component_id='matching-accuracy', component_property='value'),
     State(component_id='boolean-options', component_property='value')])
def update_results(n_clicks: int,
                   input_file: Optional[str], target_column: str,
                   model_name: str, custom_model: str, update_model: List[str],
                   tokenizer_type: str, clustering_algorithm: str,
                   keywords_extraction: str, matching_accuracy: float,
                   boolean_options: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:

    if n_clicks == 0 or not input_file or not target_column:
        return None, None, None, None, "Submit"
    if model_name == 'custom':
        model_name = custom_model
    if not update_model and not exists(model_name):
        return None, None, None, None, "Submit"

    dataframe = parse_input_file(input_file)

    options = {
        'add_placeholder': False,
        'dimensionality_reduction': False,
    }
    for option in boolean_options:
        options[option] = True

    result = execute_pipeline(dataframe, target_column,
                              model_name, bool(update_model),
                              tokenizer_type, clustering_algorithm,
                              keywords_extraction, options,
                              matching_accuracy)

    groups: pd.DataFrame = result.groups

    knee_data = None
    if clustering_algorithm == 'dbscan':
        knee_data = dumps(result.clusters.knee_data)

    if clustering_algorithm != "similarity":
        embeddings: np.ndarray = result.vectors.sent2vec
        jsoned_embeddings = pd.DataFrame(embeddings).to_json(orient='split')
    else:
        jsoned_embeddings = None

    return result.result.to_json(), groups.to_json(), jsoned_embeddings, knee_data, "Submit"


@app.callback(
    Output(component_id='results-table', component_property='children'),
    [Input('results-storage', 'children')])
def update_table(stored_results: str) -> Optional[html.Table]:
    if not stored_results:
        return None
    result: pd.DataFrame = pd.read_json(stored_results)  # type: ignore
    return generate_table(result, columns=['cluster_size', 'pattern', 'common_phrases'])


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

    clusters: Dict[int, Cluster] = {}
    for i, row in groups.iterrows():
        label: int = row['cluster']
        pattern = row['pattern'].replace("; ", ";<br>")
        if label not in clusters:
            clusters[label] = Cluster(
                x=[embeddings[i, 0]],
                y=[embeddings[i, 1]],
                hover_text=[pattern],
                size=[row['cluster_size']],
                marker_size=[log(row['cluster_size'])]
            )
        else:
            clusters[label].x.append(embeddings[i, 0])
            clusters[label].y.append(embeddings[i, 1])
            clusters[label].hover_text.append(pattern)
            clusters[label].size.append(row['cluster_size'])
            clusters[label].marker_size.append(log(row['cluster_size']))

    noise = clusters.pop(-1, Cluster([], [], [], [], []))
    clusters = {i + 1: cluster for i, cluster in enumerate(sorted(clusters.values(), reverse=True, key=lambda c: sum(c.size)))}

    max_size = max([max(cluster.marker_size) for cluster in clusters.values()] + noise.marker_size)
    noise.marker_size = [max(7, size * 25 / max_size) for size in noise.marker_size]

    for label, cluster in clusters.copy().items():
        cluster.marker_size = [max(7, size * 25 / max_size) for size in cluster.marker_size]

        if sum(cluster.size) <= noise_threshold:
            noise.x.extend(cluster.x)
            noise.y.extend(cluster.y)
            noise.size.extend(cluster.size)
            noise.hover_text.extend(cluster.hover_text)
            noise.marker_size.extend(cluster.marker_size)
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
            marker_line_width=1,
            marker_size=cluster.marker_size,
            opacity=.8,
            text=cluster.hover_text,
        ))

    fig.add_trace(Scatter(
        x=noise.x, y=noise.y,
        name="Noise",
        marker_line_width=1,
        marker_size=noise.marker_size,
        marker_color='gray',
        opacity=.8,
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
        line_color="blue"
    )

    for knee in knee_data['knees']:
        fig.add_vline(
            x=knee,
            line_color='black',
            line_dash='dash'
        )

    fig.add_vline(
        x=knee_data['chosen_knee'],
        annotation_text="Chosen epsilon value",
        line_color='green',
        line_dash='dash'
    )

    fig.update_xaxes(title_text='Number of points')
    fig.update_yaxes(title_text='Average k-neighbours distance')

    return dcc.Graph(figure=fig, responsive=True, style={'height': '90vh'})


@app.callback(
    [Output('parameters-layout', 'style'),
     Output('results-layout', 'style'),
     Output('knee-graph-layout', 'style')],
    [Input('url', 'pathname')]
)
def display_page(pathname):
    # The order in this dictionary should be the same as callback outputs
    routing_table = {
        '/': 'parameters-layout',
        '/results': 'results-layout',
        '/knee-graph': 'knee-graph-layout'
    }
    display_styles = [None if routing_table[pathname] == page else {'display': 'none'} for page in routing_table.values()]
    return tuple(display_styles)


@app.callback(
    [Output('results-nav-item', 'style'),
     Output('knee-graph-nav-item', 'style')],
    [Input('results-table', 'children'),
     Input('knee-graph', 'children')]
)
def hide_nav_items(table, knee_graph):
    results_style = {'display': 'none'}
    knee_graph_style = {'display': 'none'}
    if table:
        results_style = None
    if knee_graph:
        knee_graph_style = None
    return results_style, knee_graph_style


@app.callback(
    Output('custom-model', 'style'),
    [Input('model-file', 'value')]
)
def custom_model_form_visibility(model_name):
    if model_name == 'custom':
        return None
    return {'display': 'none'}

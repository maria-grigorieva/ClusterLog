import numpy as np
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

from os.path import exists
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple
from dash.dependencies import Input, Output, State

from webapp.process import execute_pipeline
from webapp.utility import parse_input_file, generate_table
from webapp.app import app


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
    Output(component_id='results-storage', component_property='children'),
    Output(component_id='groups-storage', component_property='children'),
    Output(component_id='embeddings-storage', component_property='children'),
    [Input(component_id='submit-button-state', component_property='n_clicks')],
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
def update_results(n_clicks: int,
                   input_file: Optional[str], target_column: str,
                   model_name: str, update_model: List[str],
                   tokenizer_type: str, clustering_algorithm: str,
                   keywords_extraction: str, threshold: int,
                   matching_accuracy: float, boolean_options: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:

    if n_clicks == 0 or not input_file or not target_column:
        return None, None, None
    if not update_model and not exists(model_name):
        return None, None, None

    dataframe = parse_input_file(input_file)

    options = {
        'add_placeholder': False,
        'dimensionality_reduction': False,
        'categorization': False
    }
    for option in boolean_options:
        options[option] = True

    result = execute_pipeline(dataframe, target_column,
                              model_name, bool(update_model),
                              tokenizer_type, clustering_algorithm,
                              keywords_extraction, options,
                              threshold, matching_accuracy)

    groups: pd.DataFrame = result.groups

    if clustering_algorithm != "similarity":
        embeddings: np.ndarray = result.vectors.sent2vec
        jsoned_embeddings = pd.DataFrame(embeddings).to_json(orient='split')
    else:
        jsoned_embeddings = None

    return result.result.to_json(), groups.to_json(), jsoned_embeddings


@app.callback(
    Output(component_id='results-table', component_property='children'),
    [Input('results-storage', 'children')])
def update_table(stored_results: str) -> Optional[html.Table]:
    if not stored_results:
        return None
    result = pd.read_json(stored_results)
    return generate_table(result, columns=['cluster_size', 'pattern', 'common_phrases'])


@app.callback(
    Output(component_id='results-graph', component_property='children'),
    [Input('groups-storage', 'children'),
     Input('embeddings-storage', 'children')])
def update_graph(stored_groups: str, stored_embeddings: str) -> Optional[dcc.Graph]:
    if not stored_embeddings or not stored_groups:
        return None

    groups: pd.DataFrame = pd.read_json(stored_groups)
    embeddings = pd.read_json(stored_embeddings, orient='split').to_numpy()

    tsne = TSNE(perplexity=25, random_state=42, verbose=False)
    embeddings = tsne.fit_transform(embeddings)

    cluster_x, cluster_y, cluster_labels, cluster_titles = [], [], [], []
    for i, row in groups.iterrows():
        cluster_x.append(embeddings[i, 0])
        cluster_y.append(embeddings[i, 1])
        cluster_labels.append(row['cluster'])
        cluster_titles.append(f"Cluster №{row['cluster']}:<br>" + row['pattern'].replace("; ", ";<br>"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cluster_x, y=cluster_y,
        name="Clusters",
        marker_line_width=1,
        marker_size=8,
        opacity=.8,
        marker_color=cluster_labels,
        text=cluster_titles
    ))

    fig.update_traces(mode="markers", hoverinfo="text")
    fig.update_layout(title="Log message clusters",
                      hovermode="closest",
                      xaxis={"visible": False},
                      yaxis={"visible": False})

    return dcc.Graph(figure=fig, style={'width': '80vw', 'height': '100vh'})
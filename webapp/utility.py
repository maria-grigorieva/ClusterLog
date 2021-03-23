import pandas as pd
import dash_html_components as html
from dash_bootstrap_components import Table

from io import StringIO
from base64 import b64decode
from collections.abc import Sequence
from typing import List, Optional
from tempfile import NamedTemporaryFile

from webapp.app import CUSTOM_MODEL_DIR


def generate_table(dataframe: pd.DataFrame, columns: Optional[List[str]] = None, max_rows: Optional[int] = None) -> Table:
    if columns is None:
        columns = [col for col in dataframe.columns]
    if max_rows is None or max_rows > len(dataframe):
        max_rows = len(dataframe)

    def format_item(item):
        if isinstance(item, Sequence):
            return html.Ul([
                html.Li(x) for x in item
            ])
        return item

    column_names = [col.replace('_', ' ').title() for col in columns]

    return Table(
        children=[
            html.Thead(
                html.Tr([html.Th(col) for col in column_names])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(format_item(dataframe.iloc[i][col])) for col in columns
                ]) for i in range(max_rows)
            ])
        ],
        bordered=True, responsive='sm', striped=True
    )


def parse_input_file(content: str) -> pd.DataFrame:
    _, content_string = content.split(',')
    decoded = b64decode(content_string)
    df = pd.read_csv(StringIO(decoded.decode('utf-8')))
    return df  # type: ignore


def parse_model_file(content: str) -> str:
    _, content_string = content.split(',')
    decoded = b64decode(content_string)
    with NamedTemporaryFile(mode='w+b', dir=CUSTOM_MODEL_DIR, delete=False, suffix='.model') as model_file:
        model_file.write(decoded)
    return model_file.name

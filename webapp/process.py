import pandas as pd

from typing import Dict
from os.path import exists

from clusterlogs.pipeline import Chain


def execute_pipeline(dataframe: pd.DataFrame, target_column: str,
                     model_name: str, update_model: bool, tokenizer_type: str,
                     clustering_algorithm: str, keywords_extraction: str,
                     options: Dict[str, bool], threshold: int, matching_accuracy: float) -> Chain:
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
    return cluster

import pandas as pd

from typing import Dict, Any

from clusterlogs.pipeline import Chain


def execute_pipeline(dataframe: pd.DataFrame, target_column: str,
                     model_name: str, model_usage_mode, tokenizer_type: str,
                     vectorization_type: str, clustering_algorithm: str,
                     clustering_parameters: Dict[str, Any], keywords_extraction: str,
                     options: Dict[str, bool], matching_accuracy: float,
                     word2vec_parameters: Dict[str, int]) -> Chain:
    cluster = Chain(dataframe, target_column,
                    vectorization_method=vectorization_type,
                    model_name=model_name, mode=model_usage_mode,
                    add_placeholder=options['add_placeholder'],
                    dimensionality_reduction=options['dimensionality_reduction'],
                    matching_accuracy=matching_accuracy,
                    output_type=None,
                    tokenizer_type=tokenizer_type,
                    clustering_type=clustering_algorithm,
                    clustering_parameters=clustering_parameters,
                    cleaning_patterns=None,
                    keywords_extraction=keywords_extraction)
    cluster.set_cluster_settings(word2vec_parameters)
    cluster.process()
    return cluster

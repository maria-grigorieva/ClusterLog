import editdistance
from typing import Sequence, Iterable, Hashable, List, Optional


T = Iterable[Hashable]


def levenshtein_similarity(a: Sequence[T], b: Sequence[T]) -> float:
    return 1 - editdistance.eval(a, b) / max(len(a), len(b))


def levenshtein_similarity_1_to_n(many: Sequence[Sequence[T]], single: Optional[Sequence[T]] = None) -> List[float]:
    if single is None:
        single, many = many[0], many[1:]
    if len(many) == 0:
        return 1
    return [levenshtein_similarity(single, item) for item in many]
import pke
import RAKE
import spacy
import pytextrank
from rake_nltk import Rake, Metric
from functools import partial
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from .data_preparation import clean_messages


def extract_common_phrases(pattern, algorithm):
    dispatch = {
        "RAKE": _extract_common_phrases_rake,
        "pyTextRank": _extract_common_phrases_pytextrank,
        "rake_nltk": _extract_common_phrases_rake_nltk,
        'lda': _extract_common_phrases_lda,

        "tfidf": partial(_extract_common_phrases_pke, algorithm="tfidf"),
        "KPMiner": partial(_extract_common_phrases_pke, algorithm="KPMiner"),
        "YAKE": partial(_extract_common_phrases_pke, algorithm="YAKE"),
        "TextRank": partial(_extract_common_phrases_pke, algorithm="TextRank"),
        "SingleRank": partial(_extract_common_phrases_pke, algorithm="SingleRank"),
        "TopicRank": partial(_extract_common_phrases_pke, algorithm="TopicRank"),
        "TopicalPageRank": partial(_extract_common_phrases_pke, algorithm="TopicalPageRank"),
        "PositionRank": partial(_extract_common_phrases_pke, algorithm="PositionRank"),
        "MultipartiteRank": partial(_extract_common_phrases_pke, algorithm="MultipartiteRank"),
        "Kea": partial(_extract_common_phrases_pke, algorithm="Kea"),
        "WINGNUS": partial(_extract_common_phrases_pke, algorithm="WINGNUS"),
    }
    try:
        return dispatch[algorithm](pattern)
    except KeyError as key:
        raise KeyError(f"Invalid keyword extraction method name: {key}! Available methods are: {tuple(dispatch.keys())}")


def _extract_common_phrases_rake(pattern):
    text = '. '.join(clean_messages(pattern))
    Rake = RAKE.Rake(RAKE.GoogleSearchStopList())
    phrases = sorted(Rake.run(text, minFrequency=1, minCharacters=3, maxWords=5),
                     key=lambda x: x[1], reverse=True)
    if len(phrases) == 0:
        return [text]
    else:
        return [item[0] for item in phrases]


def _extract_common_phrases_pytextrank(pattern):
    text = '. '.join(clean_messages(pattern))
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")
    # nlp = en_core_web_sm.load()

    # add PyTextRank to the spaCy pipeline
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    doc = nlp(text)

    phrases = []
    # examine the top-ranked phrases in the document
    for p in doc._.phrases:
        phrases.append(str(p))
    return phrases


def _extract_common_phrases_rake_nltk(pattern):
    text = '. '.join(clean_messages(pattern))
    r = Rake(min_length=2, max_length=5, ranking_metric=Metric.WORD_DEGREE)
    # Uses stopwords for english from NLTK, and all puntuation characters.
    # r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
    # r = Rake(ranking_metric=Metric.WORD_DEGREE)
    # r = Rake(ranking_metric=Metric.WORD_FREQUENCY)
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()  # To get keyword phrases ranked highest to lowest.


def _extract_common_phrases_pke(pattern, algorithm):
    dispatch = {
        "tfidf": pke.unsupervised.TfIdf,
        "KPMiner": pke.unsupervised.KPMiner,
        "YAKE": pke.unsupervised.YAKE,
        "TextRank": pke.unsupervised.TextRank,
        "SingleRank": pke.unsupervised.SingleRank,
        "TopicRank": pke.unsupervised.TopicRank,
        "TopicalPageRank": pke.unsupervised.TopicalPageRank,
        "PositionRank": pke.unsupervised.PositionRank,
        "MultipartiteRank": pke.unsupervised.MultipartiteRank,
        "Kea": pke.supervised.Kea,
        "WINGNUS": pke.supervised.WINGNUS
    }
    extractor = dispatch[algorithm]()
    extractor.load_document(input='. '.join(clean_messages(pattern)), language='en')

    selection_arguments = {
        "tfidf": {"n": 3, "stoplist": None},
        "KPMiner": {"lasf": 3, "cutoff": 400, "stoplist": None},
        "YAKE": {"n": 3, "stoplist": None},
        "TextRank": {"pos": None},
        "SingleRank": {"pos": None},
        "TopicRank": {"pos": None, "stoplist": None},
        "TopicalPageRank": {"grammar": None},
        "PositionRank": {"grammar": None, "maximum_word_number": 3},
        "MultipartiteRank": {"pos": None, "stoplist": None},
        "Kea": {"stoplist": None},
        "WINGNUS": {"grammar": None}
    }
    extractor.candidate_selection(**selection_arguments[algorithm])

    weighting_arguments = {
        "tfidf": {"df": None},
        "KPMiner": {"df": None, "sigma": 3.0, "alpha": 2.3},
        "YAKE": {"window": 2, "stoplist": None, "use_stems": False},
        "TextRank": {"window": 2, "pos": None, "top_percent": None, "normalized": False},
        "SingleRank": {"window": 2, "pos": None, "normalized": False},
        "TopicRank": {"threshold": 0.74, "method": 'average', "heuristic": None},
        "TopicalPageRank": {"window": 10, "pos": None, "lda_model": None, "stoplist": None, "normalized": False},
        "PositionRank": {"window": 2, "pos": None, "normalized": False},
        "MultipartiteRank": {"threshold": 0.74, "method": 'average', "alpha": 1.1},
        "Kea": {"model_file": None, "df": None},
        "WINGNUS": {"model_file": None, "df": None}
    }
    extractor.candidate_weighting(**weighting_arguments[algorithm])

    keyphrases = extractor.get_n_best(n=10)
    return [item[0] for item in keyphrases]


def _extract_common_phrases_lda(pattern):
    vectorizer = CountVectorizer(stop_words='english',
                                 lowercase=True,
                                 token_pattern='[-a-zA-Z][-a-zA-Z]{2,}')
    vectorized_data = vectorizer.fit_transform(pattern)
    lda = LatentDirichletAllocation(n_components=20, max_iter=10,
                                    learning_method='online',
                                    verbose=False, random_state=42)
    lda.fit(vectorized_data)

    n_topics_to_use = 10
    current_words = set()
    keywords = []

    for topic in lda.components_:
        words = [(vectorizer.get_feature_names()[i], topic[i])
                 for i in topic.argsort()[:-n_topics_to_use - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.add(word[0])

    keywords.sort(key=lambda x: x[1], reverse=True)
    return [keyword[0] for keyword in keywords][:10]

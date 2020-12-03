import RAKE
import spacy
import pytextrank
from rake_nltk import Rake, Metric


def extract_common_phrases(text, algorithm):
    dispatch = {
        "RAKE": _extract_common_phrases_rake,
        "pyTextRank": _extract_common_phrases_pytextrank,
        "rake_nltk": _extract_common_phrases_rake_nltk
    }
    return dispatch[algorithm](text)


def _extract_common_phrases_rake(text):
    stoplist = ['the','with','a','an','but','of','on','to','all','has','have','been','for','in','it','its','itself',
                'this','that','those','these','is','are','were','was','be','being','having','had','does','did','doing',
                'and','if','about','again','then','so','too','cern','cms','atlas','by','srm','ifce', 'err','error']
    Rake = RAKE.Rake(stoplist)#Rake = RAKE.Rake(RAKE.GoogleSearchStopList())
    phrases = sorted(Rake.run(text, minFrequency=1, minCharacters=3, maxWords=10),
                     key=lambda x: x[1], reverse=True)
    if len(phrases) == 0:
        return [text]
    else:
        return [item[0] for item in phrases]


def _extract_common_phrases_pytextrank(text):
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")

    # add PyTextRank to the spaCy pipeline
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    doc = nlp(text)

    phrases = []
    # examine the top-ranked phrases in the document
    for p in doc._.phrases:
        phrases.append(str(p))
    return phrases


def _extract_common_phrases_rake_nltk(text):
    r = Rake(min_length=2, max_length=5, ranking_metric=Metric.WORD_FREQUENCY)  # Uses stopwords for english from NLTK, and all puntuation characters.
    # r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
    # r = Rake(ranking_metric=Metric.WORD_DEGREE)
    # r = Rake(ranking_metric=Metric.WORD_FREQUENCY)
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()  # To get keyword phrases ranked highest to lowest.

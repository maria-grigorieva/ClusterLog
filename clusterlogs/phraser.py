import RAKE
import spacy
import pytextrank


def extract_common_phrases(text, algorithm):
    dispatch = {
        "RAKE": _extract_common_phrases_rake,
        "pyTextRank": _extract_common_phrases_pytextrank
    }
    return dispatch[algorithm](text)


def _extract_common_phrases_rake(text):
    Rake = RAKE.Rake(RAKE.GoogleSearchStopList())  # Google is the smallest built-in list
    phrases = Rake.run(text, minFrequency=1, minCharacters=3, maxWords=5)
    # Rake.run() already returns sorted pairs, this shouldn't be necessary
    phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
    if not phrases:
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
        phrases.append(p)
    return phrases

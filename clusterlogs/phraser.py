import RAKE
import spacy
import pytextrank


class phraser:

    def __init__(self, text, type='RAKE'):
        self.text = text
        self.type = type


    def extract_common_phrases(self):
        if self.type == 'RAKE':
            Rake = RAKE.Rake(RAKE.GoogleSearchStopList())
            phrases = sorted(Rake.run(self.text, minFrequency=1, minCharacters=3, maxWords=5),
                             key=lambda x: x[1], reverse=True)
            if len(phrases) == 0:
                return [self.text]
            else:
                return [item[0] for item in phrases]

        if self.type == 'pyTextRank':
            # load a spaCy model, depending on language, scale, etc.
            nlp = spacy.load("en_core_web_sm")

            # add PyTextRank to the spaCy pipeline
            tr = pytextrank.TextRank()
            nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

            doc = nlp(self.text)

            phrases = []

            # examine the top-ranked phrases in the document
            for p in doc._.phrases:
                phrases.append(p)
            return phrases





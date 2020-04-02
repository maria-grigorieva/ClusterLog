import RAKE

class phraser:

    def __init__(self, text):
        self.text = text


    def extract_common_phrases(self):
        Rake = RAKE.Rake(RAKE.GoogleSearchStopList())
        phrases = sorted(Rake.run(self.text, minFrequency=1, minCharacters=3, maxWords=5),
                         key=lambda x: x[1], reverse=True)
        if len(phrases) == 0:
            return self.text
        else:
            return [item for item in phrases]


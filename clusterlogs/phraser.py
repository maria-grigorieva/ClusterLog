import RAKE

class phraser:

    def __init__(self, text):
        self.text = text


    def extract_common_phrases(self):
        Rake = RAKE.Rake(RAKE.SmartStopList())
        phrases = sorted(Rake.run(self.text), key=lambda x: x[1], reverse=True)
        return [item[0] for item in phrases]


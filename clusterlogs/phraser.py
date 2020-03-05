import RAKE

class phraser:

    def __init__(self, text):
        self.text = text


    def extract_common_phrases(self):
        Rake = RAKE.Rake(RAKE.SmartStopList())
        return sorted(Rake.run(self.text), key=lambda x: x[1], reverse=True)


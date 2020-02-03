import re

class Regex(object):

    def __init__(self, messages):
        self.messages = messages
        self.messages_cleaned = None

    def process(self):
        """
        :return:
        """
        self.messages_cleaned = [0] * len(self.messages)
        for idx, item in enumerate(self.messages):
            try:
                item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', '*', item)
                self.messages_cleaned[idx] = item
            except Exception as e:
                print(item)
        return self.messages_cleaned

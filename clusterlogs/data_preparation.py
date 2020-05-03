import re
import pprint

def clean_messages(messages):
    messages_cleaned = []
    for item in messages:
        item = re.sub(r'\S+\.\S+', ' ', item)  # any URL
        # item = re.sub(r'(/[\w\./]*\s?)', ' ', item)
        # item = re.sub(r'([a-zA-Z0-9]+[_]+\S+)', ' ', item)
        # item = re.sub(r'(\d+)', ' ', item)
        item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', item)
        item = re.sub(r'[^\w\s]', ' ', item) # removes all punctuation
        item = re.sub(r' +', r' ', item)
        messages_cleaned.append(item)
    # pprint.pprint(messages_cleaned)
    return messages_cleaned


def alpha_cleaning(messages):
    messages_cleaned = []
    for item in messages:
        item = re.sub(r'\S+\.\S+', ' ', item)  # any URL
        item = re.sub(r'(\d+)', ' ', item)
        item = re.sub(r'[^\w\s]', ' ', item)
        item = re.sub(r' +', r' ', item)
        messages_cleaned.append(item)
    return messages_cleaned
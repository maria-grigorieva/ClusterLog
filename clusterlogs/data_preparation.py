import re


def clean_messages(messages):
    """
    :return:
    """
    messages_cleaned = [0] * len(messages)
    for idx, item in enumerate(messages):
        item = re.sub(r'[^ ]+\.[^ ]+', ' ', item) # any URL
        item = re.sub(r'(/[\w\./]*[\s]?)', ' ', item)
        item = re.sub(r'([a-zA-Z0-9]+[_]+[\S]+)', ' ', item)
        item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', item)
        item = re.sub(r'[^\w\s]', ' ', item)
        item = re.sub(' +', ' ', item)
        messages_cleaned[idx] =  item
    return messages_cleaned

import re


def clean_messages(messages):
    messages_cleaned = []
    for item in messages:
        # Removes anything other than whitespace that contains '.' inside
        # Will not remove words that start or end in '.'
        item = re.sub(r'\S+\.\S+', ' ', item)
        # Removes words that contain at least one digit inside but not in the last place
        item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', item)
        # Changes every symbol other than letters, whitespace and '_' to ' '
        item = re.sub(r'[^\w\s]', ' ', item)
        # Remove duplicate spaces
        item = re.sub(r' +', r' ', item)
        messages_cleaned.append(item.lower())
    return messages_cleaned

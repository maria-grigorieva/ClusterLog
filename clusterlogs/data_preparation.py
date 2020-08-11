import re
# import pprint


def clean_messages(messages):
    messages_cleaned = []
    for item in messages:
        # Removes anything other than whitespace that contains '.' inside
        # Will not remove words that start or end in '.'
        item = re.sub(r'\S+\.\S+', ' ', item)
        # item = re.sub(r'(/[\w\./]*\s?)', ' ', item)
        # item = re.sub(r'([a-zA-Z0-9]+[_]+\S+)', ' ', item)
        # item = re.sub(r'(\d+)', ' ', item)
        # Removes words that contain at least one digit inside but not in the last place
        item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', item)
        # Changes every symbol other than letters, whitespace and '_' to ' '
        item = re.sub(r'[^\w\s]', ' ', item)
        item = re.sub(r' +', r' ', item)
        messages_cleaned.append(item.lower())
        # print(item)
    # pprint.pprint(messages_cleaned)
    return messages_cleaned


# def alpha_cleaning(messages):
#     messages_cleaned = []
#     for item in messages:
#         # Removes anything other than whitespace that contains '.' inside
#         # Will not remove words that start or end in '.'
#         item = re.sub(r'\S+\.\S+', ' ', item)
#         # Removes digits. May lead to words being split in 2.
#         # Should not be necessary with the next line
#         item = re.sub(r'(\d+)', ' ', item)
#         # Changes every symbol other than letters, whitespace and '_' to ' '
#         item = re.sub(r'[^\w\s]', ' ', item)
#         item = re.sub(r' +', r' ', item)
#         messages_cleaned.append(item)
#     return messages_cleaned

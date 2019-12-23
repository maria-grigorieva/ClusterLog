import re
from kneed import KneeLocator
import matplotlib.pyplot as plt

_uid = r'[0-9a-zA-Z]{12,128}'
_line_number = r'(at line[:]*\s*\d+)'
_uuid = r'[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}'
_url = r'(http[s]|root|srm|file)*:(//|/)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
_filepath = "(/[a-zA-Z\./]*[\s]?)"
path_regex = re.compile(r'(\b\w+://)\S+(?=\s)')
file_regex = re.compile(r'(\b[f|F]ile( exists)?:?\s?)/\S+(?=\s)')
py_regex = re.compile(r'/?\b[-./_a-zA-Z0-9]+\.py\b')
long_regex = re.compile(r'[-./_a-zA-Z0-9]{25,}')

def remove_whitespaces(sentence):
    """
    Some error messages has multiple spaces, so we change it to one space.
    :param sentence:
    :return:
    """
    return " ".join(sentence.split())


def cleaner(messages):
    """
    Clear error messages from unnecessary data:
    - UID/UUID in file paths
    - line numbers - as an example "error at line number ..."
    Removed parts of text are substituted with titles
    :return:
    """

    for idx, item in enumerate(messages):
        item = re.sub(_line_number, "at line LINE_NUMBER", item)
        item = re.sub(_uid, "UID", item)
        item = re.sub(_uuid, "UUID", item)
        item = re.sub(_url, "URL", item)
        item = re.sub("\d+", "NUM", item)
        item = substitute_path(item)
        messages[idx] = remove_whitespaces(item)
    return messages


def substitute_path(string):
    string = path_regex.sub(r'\1<PATH>', string)
    string = py_regex.sub(r'<FILE.py>', string)
    string = file_regex.sub(r'\1<FILE>', string)
    string = long_regex.sub(r'LONG', string)
    return string

def distance_curve(distances, mode='show'):
    """
    Save distance curve with knee candidates in file.
    :param distances:
    :param mode: show | save
    :return:
    """
    sensitivity = [1, 3, 5, 10, 100, 150]
    knees = []
    y = list(range(len(distances)))
    for s in sensitivity:
        kl = KneeLocator(distances, y, S=s)
        knees.append(kl.knee)

    plt.style.use('ggplot');
    plt.figure(figsize=(10, 10))
    plt.plot(distances, y)
    colors = ['r', 'g', 'k', 'm', 'c', 'b', 'y']
    for k, c, s in zip(knees, colors, sensitivity):
        plt.vlines(k, 0, len(distances), linestyles='--', colors=c, label=f'S = {s}')
        plt.legend()
        if mode == 'show':
            plt.show()
        else:
            plt.savefig("distance_curve.png")
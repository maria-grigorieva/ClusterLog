from re import sub
from kneed import KneeLocator
import matplotlib.pyplot as plt

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
    _uid = r'[0-9a-zA-Z]{12,128}'
    _line_number = r'(at line[:]*\s*\d+)'
    _uuid = r'[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}'

    for idx, item in enumerate(messages):
        _cleaned = sub(_line_number, "at line LINE_NUMBER", item)
        _cleaned = sub(_uid, "UID", _cleaned)
        _cleaned = sub(_uuid, "UUID", _cleaned)
        messages[idx] = remove_whitespaces(_cleaned)
    return messages


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
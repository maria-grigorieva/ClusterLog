from kneed import KneeLocator
import matplotlib.pyplot as plt

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
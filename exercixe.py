import numpy as np
import matplotlib.pyplot as plt

experiences = {f"experience_{i}": np.random.randint(
    0, 10, 10) for i in range(4)}

"""
    It takes a dictionary of lists and plots each list as a subplot.

    :param experiences: a dictionary of lists, where each list is a list of numbers
    """


def make_experience(experiences):
    print(experiences)
    for i, k in enumerate(experiences):
        experience = experiences[k]
        plt.subplot(len(experiences), 1, i + 1)
        plt.plot(experience)
    plt.show()


make_experience(experiences)

"""
    Author: Pascal Pilz, k12111234
    SS 2022
    KO Mathematics for AI II, 324.813
    https://github.com/pilzpascal/KO_Mathematics_for_AI_II

    This is an implementation of the Gram-Schmidt process for orthonormalizing vector basis.

    This implementation uses numpy. For more details on the topic see the accompanying GitHub repository and the slides.
"""


import numpy as np
from numpy.linalg import norm
from numpy import dot


def gram_schmidt_process_numpy(s: list) -> list:
    """
    Generates am orthonormal base spanning the same space as the input-base.

    :param list s: the base spanning the desired space

    :return: the generated orthonormal base

    :raises:
    """
    # A little helper function to keep it cleaner
    def normalize(vector: np.ndarray) -> np.ndarray:
        return vector / norm(vector)

    s = np.array(s)

    y_k = list()
    for x_k in s:
        element = 0
        for y_i in y_k or [0]:
            element += dot(x_k, y_i) * y_i
        y_k.append(normalize(x_k - element))
    return [list(y_i) for y_i in y_k]

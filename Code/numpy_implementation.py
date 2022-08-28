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

from utils import get_input, print_preamble, draw_vectors


def check_linear_independence(s: list) -> bool:
    """
    Checks if a list of given vectors are linearly independent using singular value decomposition.

            Parameters:
                    s (list): list of vectors

            Returns:
                True if the vectors are linearly independent, False else.
    """
    # This way of determining linear independence was found at
    # https://kitchingroup.cheme.cmu.edu/blog/2013/03/01/Determining-linear-independence-of-a-set-of-vectors/
    length = len(s)
    stacked_matrix = np.row_stack(s)
    eps = np.finfo(norm(stacked_matrix).dtype).eps
    tolerance = max(eps * np.array(stacked_matrix.shape))
    _, s, _ = np.linalg.svd(stacked_matrix)
    if np.sum(s > tolerance) < length:
        return False
    return True


def check_properties_base(s: list) -> None:
    """
    Checks if the given list of vectors fulfills the required properties, namely
        - the number of vectors must be equal to the length of the individual vectors
        - all the element are of equal length
        - all vectors must be linearly independent

            Parameters:
                    s (list): list of vectors

            Returns:
                None

            Raises:
                ValueError
    """
    length = len(s)
    for j in range(length):
        # We don't care to orthonormalize basis of one dimensional real vector spaces
        if len(s[j]) < 2:
            raise ValueError(f"An element of the base is shorter than the minimum required length.")
        # All elements of a base need to be of equal length, i.e., of the same dimension
        if len(s[0]) != len(s[j]):
            raise ValueError(f"Two entered vectors are not of equal length: {s[0]} and {s[j]}.")
    # The number of elements of a base is equal to dimension of the space they span
    if length > len(s[0]):
        raise ValueError("The number of elements in the base is greater than the length of the individual elements.")
    if not check_linear_independence(s):
        raise ValueError("Entered vectors are not linearly independent.")


def gram_schmidt_process_numpy(s: list) -> list:
    """
    Generates am orthonormal base spanning the same space as the input-base.

            Parameters:
                    s (list): the base spanning the desired space

            Returns:
                base (list): the generated orthonormal base
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


if __name__ == "__main__":
    print_preamble()

    print()
    print("="*105)
    print()

    input_base = get_input()
    check_properties_base(input_base)
    print(f"The entered base is {input_base}")

    print()
    print("="*105)
    print()

    output_base = gram_schmidt_process_numpy(input_base)
    print(f"The output of the Gram-Schmidt process is: {output_base}")
    draw_vectors(input_base, output_base)

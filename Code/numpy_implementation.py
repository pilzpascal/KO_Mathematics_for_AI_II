""""
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
    A = np.row_stack(s)
    eps = np.finfo(norm(A).dtype).eps
    TOLERANCE = max(eps * np.array(A.shape))
    U, s, V = np.linalg.svd(A)
    if np.sum(s > TOLERANCE) < length:
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


def check_valid_input(string: str) -> None:
    """
    Checks if a given string is a number. Able to identify negative numbers and numbers containing at most one dot.

            Parameters:
                    string (str): string to be checked

            Returns:
                None

            Raises:
                ValueError
    """
    # This expression is required to ensure that negative and decimal numbers are accepted
    if not string.lstrip("-").replace(".", "", 1).isnumeric():
        raise ValueError(f"Invalid input: {string}.")


def get_input() -> list:
    list_of_vectors = [[]]
    while True:
        last_entered = input("enter value: ")
        if last_entered == "X":
            print(f"Base entered so far: {list_of_vectors}.")
            check_properties_base(list_of_vectors)
            list_of_vectors.append(list())
            continue
        if last_entered == "XX":
            check_properties_base(list_of_vectors)
            print(f"The entered base is:{list_of_vectors}")
            break
        check_valid_input(last_entered)
        list_of_vectors[-1].append(float(last_entered))
    return list_of_vectors


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
    print("=" * 105)
    print()
    print("The base can be of any real vector space with a dimension greater or equal than 2, i.e, R^2, R^3, ...")
    print()
    print("The base is to be entered as follows: each element of the vector is entered separately, confirmed by the")
    print("'enter'-key. If 'X' gets entered, a vector is concluded and the next one can be entered. Once the base is")
    print("fully entered, 'XX' should be entered. Only valid bases are accepted, that is:")
    print("    - all vectors must be linearly independent,")
    print("    - all vectors must be of equal length,")
    print("    - the number of vectors must be equal to the length of the individual vectors.")
    print("Decimal numbers are entered with a dot.")
    print()
    print("Example for valid input: '1', '2', 'X', '-1', '-0.5', 'XX'")
    print()
    print("="*105)
    print()

    base = get_input()

    print(gram_schmidt_process_numpy(base))

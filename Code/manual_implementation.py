"""
    Author: Pascal Pilz, k12111234
    SS 2022
    KO Mathematics for AI II, 324.813
    https://github.com/pilzpascal/KO_Mathematics_for_AI_II

    This is an implementation of the Gram-Schmidt process for orthonormalizing vector basis.

    This implementation uses pure python. For more details on the topic see the accompanying GitHub repository and the
    slides.
"""


def inner_product(x: list, y: list) -> float:
    """
    Returns the inner product of two real valued vectors of arbitrary dimension. The two vectors must be of same length.

    :param list x: vector
    :param list y: vector

    :return: the inner product, defined as sum of element-wise multiplication, sum_{i=1}^{n}{x_i*y_i}
    """
    if not len(x) == len(y):
        raise ValueError("The two vectors need to be of the same dimension.")
    prod = 0
    for x_i, y_i in zip(x, y):
        prod += x_i * y_i
    return prod


def norm(vector: list) -> float:
    """
    Returns the norm of the given vector of R^n, defined as the square root of the inner product with itself.

    :param list vector: vector of which the norm is to be taken; can be of R^n

    :return: the inner product, defined as sum of element-wise multiplication, sum_{i=1}^{n}{x_i*y_i}

    :raises:
    """
    return inner_product(vector, vector) ** 0.5


def v_s_mul(vector: list, scalar: float) -> list:
    """
    Multiplies a given vector and a scalar.

    :param list vector: vector to be scaled
    :param float scalar: an element of the underlying field; value to be scaled by

    :return: the scaled vector

    :raises:
    """
    return [elem * scalar for elem in vector]


def v_add(*args) -> list:
    """
    Adds the given vectors component-wise.

    :param args: vectors to be added, supplied like v_add(vec1, vec2, vec3, ...)

    :return: a vector; the resulting sum

    :raises:
    """
    num = [0] * len(args[0])
    for vector in args:
        num = [num_elem + vec_elem for num_elem, vec_elem in zip(num, vector)]
    return num


def v_sub(*args) -> list:
    """
    Subtracts the given vectors component-wise. in the given order, i.e., vec1 - vec2 - vec3 ...

    :param args: vectors to be subtracted, supplied like v_sub(vec1, vec2, vec3, ...)

    :return: the resulting sum

    :raises:
    """
    if len(args) == 1:
        return args[0]
    return v_add(args[0], v_s_mul(vector=v_add(*args[1:]), scalar=-1))


def gram_schmidt_process_manual(s: list) -> list:
    """
    Generates am orthonormal base spanning the same space as the input-base.

    :param list s: the base spanning the desired space

    :return: the generated orthonormal base

    :raises:
    """

    # A little helper function to keep it cleaner
    def normalize(vector: list) -> list:
        return [elem / norm(vector) for elem in vector]

    y_k = list()
    for x_k in s:
        element = [0] * len(s[0])
        for y_i in y_k or [element]:
            element = v_add(element, v_s_mul(vector=y_i, scalar=inner_product(x_k, y_i)))
        y_k.append(normalize(v_sub(x_k, element)))
    return y_k

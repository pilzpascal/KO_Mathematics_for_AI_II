""""
    Author: Pascal Pilz, k12111234
    SS 2022
    KO Mathematics for AI II, 324.813

    This is an implementation of the Gram-Schmidt process for orthonormalizing vector basis.

    This implementation uses pure python. For more details on the topic see the accompanying GitHub repository and the
    slides.
"""


def inner_product(x: list, y: list) -> float:
    """
    Returns the inner product of two real valued vectors of arbitrary dimension. The two vectors must be of same length.

            Parameters:
                    x (list): vector
                    y (list): vector

            Returns:
                prod (float): The inner product, defined as sum of element-wise multiplication, sum_{i=1}^{n}{x_i*y_i}
    """
    if not len(x) == len(y):
        raise ValueError("The two vectors need to be of the same dimension.")
    prod = 0
    for x_i, y_i in zip(x, y):
        prod += x_i * y_i
    return prod


def norm(vector: list) -> float:
    """
    Returns the norm of the given vector, defined as the square root of the inner product with itself.

            Parameters:
                    vector (list): vector

            Returns:
                prod (float): The inner product, defined as sum of element-wise multiplication, sum_{i=1}^{n}{x_i*y_i}
    """
    return inner_product(vector, vector) ** 0.5


def check_linear_independence(s: list) -> bool:
    """
    Checks if a list of given vectors are linearly independent using the Cauchy-Schwarz inequality.

            Parameters:
                    s (list): list of vectors

            Returns:
                True if the vectors are linearly independent, False else.
    """
    # We are using the Cauchy-Schwarz inequality to check for linear dependence, as seen here
    # https: // web.archive.org / web / 20220815112144 / https: // endlesslernen.wordpress.com / 2018 / 03 / 29 /
    # linear - independent - check - in -python /
    for x in s:
        for y in s:
            if not x == y:
                if abs(abs(inner_product(x, y)) - norm(x) * norm(y)) \
                        < 1e-9 * max(abs(inner_product(x, y)), norm(x) * norm(y)):
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


def v_s_mul(vector: list, scalar: float) -> list:
    """
    Multiplies a given vector and a scalar.

            Parameters:
                    vector (list): vector to be scaled
                    scalar (float): value to be scaled my

            Returns:
                (list): the scaled vector
    """
    return [elem * scalar for elem in vector]


def v_add(*args) -> list:
    """
    Adds the given vectors component-wise.

            Parameters:
                    args: vectors to be added, supplied like v_add(vec1, vec2, vec3, ...)

            Returns:
                (list): the resulting sum
    """
    num = [0] * len(args[0])
    for vector in args:
        num = [num_elem + vec_elem for num_elem, vec_elem in zip(num, vector)]
    return num


def v_sub(*args) -> list:
    """
    Subtracts the given vectors component-wise. in the given order, i.e., vec1 - vec2 - vec3 ...

            Parameters:
                    args: vectors to be subtracted, supplied like v_sub(vec1, vec2, vec3, ...)

            Returns:
                (list): the resulting sum
    """
    if len(args) == 1:
        return args[0]
    return v_add(args[0], v_s_mul(vector=v_add(*args[1:]), scalar=-1))


def gram_schmidt_process_manual(s: list) -> list:
    """
    Generates am orthonormal base spanning the same space as the input-base.

            Parameters:
                    s (list): the base spanning the desired space

            Returns:
                base (list): the generated orthonormal base
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
    print("=" * 105)
    print()

    base = get_input()

    print(gram_schmidt_process_manual(base))

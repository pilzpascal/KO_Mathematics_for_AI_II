import matplotlib.pyplot as plt

from math import ceil, floor

from numpy.linalg import norm
from numpy import dot


def check_valid_input(string: str) -> None:
    """
    Checks if a given string is a number. Able to identify negative numbers and numbers containing at most one dot.

    :param str string: string to be checked

    :return: None

    :raises ValueError: the entered input is not in the valid range
    """
    # This expression is required to ensure that negative and decimal numbers are accepted
    if not string.lstrip("-").replace(".", "", 1).isnumeric():
        raise ValueError(f"Invalid input: {string}.")


def print_seperator() -> None:
    """
    Prints a seperator; like "\n" + 105 * "=" + "\n"

    :param:

    :return:

    :raises:
    """
    print()
    print("=" * 105)
    print()


def get_implementation() -> str:
    """
    Asks the user which implementation they want to use.
    Current available (valid inputs) are:
        - numpy
        - python

    :param:

    :return: the desired implementation

    :raises ValueError: the entered input is not in the valid range
    """
    print_seperator()

    versions = ["numpy", "n", "manual", "m"]
    print("Please choose which implementation you would like to use. This choice only affects the background")
    print("calculations, the result will be the same. Available are:")
    print(*["- " + elem for elem in versions[::2]], sep="\n")
    print()
    print(f"Enter either one of those to proceed ({versions}).")
    print()

    implementation = input("Desired implementation: ")
    if implementation not in versions:
        raise ValueError(f"The input {implementation} is not in {versions} and thus invalid. "
                         f"A frequent cause of error is an accidental space.")
    return implementation


def get_input() -> list:
    """
    Gets the input form the user via the command line. For more information on how the input should be supplied see the
    function "print_preamble". This function utilizes the function "check_valid_input" to check whether the entered
    string is a valid number.

    :param:

    :return: the list which was entered by the user

    :raises:
    """
    print_seperator()

    print("Enter the elements of the basis:")
    element = input()
    element = [float(elem) for elem in element.split()]
    basis = [element]
    dim = len(element)

    # We do this while loop so that once all elements are entered the process automatically terminates
    while len(basis) < dim:
        element = input()
        try:
            element = [float(elem) for elem in element.split()]
        except ValueError:
            print("It appears you entered something that could not converted to float.")
        else:
            basis.append(element)
    return basis


def print_preamble() -> None:
    """
    Prints a little preamble explaining how to use this script. Does nothing else.

    :param:

    :returns:

    :raises:
    """
    print_seperator()

    print("The base can be of any real vector space with a dimension greater or equal than 2, i.e, R^2, R^3, ...")
    print("The base is to be entered as follows: each element of the base (vector) is entered at a time, the elements")
    print("of the vector are separated by a SPACE and the completed vector is confirmed with the ENTER key.")
    print("The first entered vector determines the dimension of the vector base, so if for example the first entered")
    print("vector is '1 2' then the dimension will be 2, i.e., R^2, if the first element is '1 2 3 4' we have R^4.")
    print("As soon as the full basis is entered, i.e., as many vectors were given as is the dimension (length) of the")
    print("individual vectors, the input will automatically be processed.")
    print()
    print("Negative numbers have a leading '-' (minus), no space between the '-' and the number, e.g. '-1'.")
    print("Decimal numbers use a '.' (dot), no space between the '.' and the numbers, e.g. '1.14'.")
    print()
    print("Only a valid base will be accepted, that is:")
    print("    - all vectors must be linearly independent,")
    print("    - all vectors must be of equal length,")
    print("    - the number of vectors must be equal to the length of the individual vectors.")

    print_seperator()

    print("Example for valid input:")
    print("1 2")
    print("-1 -0.5")
    print()
    print("Example for valid input:")
    print("1 2 3 4")
    print("5 6 7 8")
    print("9 10 11 12")
    print("13 14 15 16")


def check_linear_independence(vectors: list) -> bool:
    """
    Checks if a list of given vectors are linearly independent using the Cauchy-Schwarz inequality.

    :param list vectors: list of vectors

    :return: True if the vectors are linearly independent, False else.

    :raises:
    """
    # We are using the Cauchy-Schwarz inequality to check for linear dependence, as seen here
    # https://web.archive.org/web/20220815112144/https://endlesslernen.wordpress.com/2018/03/29/linear-independent-check-in-python/
    for x in vectors:
        for y in vectors:
            if x is not y:
                tolerance = 1e-9 * max(abs(dot(x, y)), norm(x) * norm(y))
                if abs(abs(dot(x, y)) - norm(x) * norm(y)) < tolerance:
                    return False
    return True


def check_properties_base(s: list) -> None:
    """
    Checks if the given list of vectors fulfills the required properties, namely
        - the number of vectors must be equal to the length of the individual vectors
        - all the element are of equal length
        - all vectors must be linearly independent

    :param list s: list of vectors

    :return: None

    :raises ValueError: the entered base does not fulfill the necessary properties
    """
    length = len(s)
    for j in range(length):
        # We don't care to orthonormalize basis of one dimensional real vector spaces
        # Since a set containing one element cannot be orthonormal
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


def draw_vectors(vectors: list) -> None:
    """
    Draws two- and three-dimensional vectors using matplotlib.pyplot.quiver.

    :param list vectors: list vectors, like [[x11, x12,], ..., [xn1, xn2]] or [[x11, x12, x13], ..., [xn1, xn2, xn3]]

    :return:

    :raises ValueError: the passed elements are not of the right form
    """
    dim = len(vectors[0])
    if (dim == 2 and sum([len(elem) != 2 for elem in vectors])) \
            or (dim == 3 and sum([len(elem) != 3 for elem in vectors])):
        raise ValueError(f"Function for drawing vectors of R^2 or R^3 was called on a base of R^{dim}.")

    fig = plt.figure()

    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError(f"Drawing unction was called for vectors of dimension {dim}.")

    # All vectors we want to draw start at 0V
    origin = [[0] * len(vectors)] * dim
    x_dir = [elem[0] for elem in vectors]
    y_dir = [elem[1] for elem in vectors]
    if dim == 3:
        z_dir = [elem[2] for elem in vectors]
        plt.quiver(*origin,
                   x_dir, y_dir, z_dir,
                   color=["red", "blue", "green", "pink", "cyan", "lime"])
    else:
        plt.quiver(*origin,
                   x_dir, y_dir,
                   color=["red", "blue", "pink", "cyan"],
                   angles='xy', scale_units='xy', scale=1)

    # Getting the min and max to make the visualization a bit more pleasant and adapting to the vectors
    x_min = min([elem[0] for elem in vectors])
    x_max = max([elem[0] for elem in vectors])
    y_min = min([elem[1] for elem in vectors])
    y_max = max([elem[1] for elem in vectors])
    if dim == 3:
        z_max = max([elem[2] for elem in vectors])
        z_min = min([elem[2] for elem in vectors])
        axis_min = floor(min([x_min, y_min, z_min]) + 1 / 10 * min([x_min, y_min, z_min]))
        axis_max = ceil(max([x_max, y_max, z_max]) + 1 / 10 * max([x_max, y_max, z_max]))
        plt.xlim(left=axis_min, right=axis_max)
        plt.ylim(bottom=axis_min, top=axis_max)
        ax.set_zlim([axis_min, axis_max])
        ax.set_aspect('auto', adjustable='box')
    else:
        axis_min = floor(min([x_min, y_min]) + 1 / 10 * min([x_min, y_min]))
        axis_max = ceil(max([x_max, y_max]) + 1 / 10 * max([x_max, y_max]))
        plt.xlim(left=axis_min, right=axis_max)
        plt.ylim(bottom=axis_min, top=axis_max)
        ax.set_aspect('equal', adjustable='box')

    plt.show()

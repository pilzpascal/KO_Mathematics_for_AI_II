import matplotlib.pyplot as plt
from math import ceil, floor


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
    """
    Gets the input form the user via the command line. For more information on how the input should be supplied see the
    function "print_preamble". This function utilizes the function "check_valid_input" to check whether the entered
    string is a valid number.

            Parameters:

            Returns:
                basis (list): the entered list
    """
    print("Enter the elements of the basis:")
    element = input()
    element = [float(elem) for elem in element.split()]
    basis = [element]
    dim = len(element)

    while len(basis) < dim:
        element = input()
        try:
            element = [float(elem) for elem in element.split()]
        except ValueError:
            print("It appears you entered something that could not converted to float.")
        else:
            basis.append(element)
    return basis


def print_seperator() -> None:
    """
    Prints a seperator consisting of a new line, 105 * "=", a new line.

            Parameters:

            Returns:
    """
    print()
    print("="*105)
    print()


def print_preamble() -> None:
    """
    Prints a little preamble explaining how to use this script. Does nothing else.

            Parameters:

            Returns:
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

    print_seperator()


def draw_vectors(input_base, orthonormalized_base) -> None:
    """
    Draws two-dimensional vectors using matplotlib.pyplot.quiver

            Parameters:
                input_base (list): the base that was input by the user, given as [[x11, x12], [x21, x22]]
                orthonormalized_base (list): the output of the Gram-Schmidt process, given as [[y11, y12], [y21, y22]]

            Returns:
    """
    if not len(input_base) == 2:
        print("There is no drawing since the given vectors are not of R^2.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)

    origin = [[0, 0, 0, 0], [0, 0, 0, 0]]
    x_dir = [elem[0] for elem in input_base + orthonormalized_base]
    y_dir = [elem[1] for elem in input_base + orthonormalized_base]
    plt.quiver(*origin,
               x_dir, y_dir,
               color=["red", "blue", "pink", "cyan"],
               angles='xy', scale_units='xy', scale=1)

    x_min = min([elem[0] for elem in input_base + orthonormalized_base])
    x_max = max([elem[0] for elem in input_base + orthonormalized_base])
    y_min = min([elem[1] for elem in input_base + orthonormalized_base])
    y_max = max([elem[1] for elem in input_base + orthonormalized_base])
    axis_min = floor(min([x_min, y_min]) + 1/10 * min([x_min, y_min]))
    axis_max = ceil(max([x_max, y_max]) + 1/10 * max([x_max, y_max]))

    plt.xlim(left=axis_min, right=axis_max)
    plt.ylim(bottom=axis_min, top=axis_max)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

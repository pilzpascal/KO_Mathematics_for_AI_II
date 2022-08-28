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
                list_of_vectors (list): the entered list
    """
    list_of_vectors = [[]]
    while True:
        last_entered = input("enter value: ")
        if last_entered == "X":
            print(f"Base entered so far: {list_of_vectors}.")
            list_of_vectors.append(list())
            continue
        if last_entered == "XX":
            break
        check_valid_input(last_entered)
        list_of_vectors[-1].append(float(last_entered))
    return list_of_vectors


def print_preamble() -> None:
    """
    Prints a little preamble explaining how to use this script. Does nothing else.

            Parameters:

            Returns:
    """
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

from utils import *
from numpy_implementation import gram_schmidt_process_numpy
from manual_implementation import gram_schmidt_process_manual


def main():
    print_preamble()

    while True:
        try:
            implementation = get_implementation()
        except ValueError as exception:
            print(exception)
        else:
            break

    while True:
        try:
            input_base = get_input()
            check_properties_base(input_base)
        except ValueError as exception:
            print("It appears that the values you entered are not a valid base. The error is:")
            print(exception)
        else:
            print(f"The entered base is {input_base}")
            break

    print_seperator()

    if implementation in ["numpy", "n"]:
        print("Calculation done in numpy.")
        output_base = gram_schmidt_process_numpy(input_base)
    elif implementation in ["manual", "m"]:
        print("Calculation done in pure python.")
        output_base = gram_schmidt_process_manual(input_base)
    else:
        raise ValueError(f"Something went wrong. {implementation} was chosen as implementation.")
    print(f"The output of the Gram-Schmidt process is: {output_base}")

    if len(input_base) in [2, 3]:
        draw_vectors(input_base + output_base)
    else:
        print("There is no drawing since the given base is not of R^2 or R^3.")


if __name__ == "__main__":
    main()

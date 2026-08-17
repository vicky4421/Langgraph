# a try to get rid of typos, partially works for strings but don't work with function names while adding nodes

from enum import StrEnum

class Node(StrEnum):
    SHOW_EQUATION = 'show_equation'
    CALCULATE_DISCRIMINANT = 'calculate_discriminant'
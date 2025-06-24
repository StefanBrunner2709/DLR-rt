"""
Contains class to set up low rank structure.
"""


class LR:
    """
    Low rank class.

    Generate low rank structure using matrices U, S and V.
    """
    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V

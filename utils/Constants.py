# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Goal configuration: blank in the top-left corner, tiles in ascending order.
GOAL = (0, 1, 2, 3, 4, 5, 6, 7, 8)

# Possible moves of the blank as (move name, row change, column change).
# The list order is also the order in which children are expanded.
MOVES = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]
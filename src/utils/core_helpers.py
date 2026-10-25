# =====================================================================
#    CORE HELPERS
#    Problem definition: input parsing, goal test, solvability test,
#    successor function and path reconstruction.
# =====================================================================

from utils.Constants import *

def parse_state(text):
    """
    Convert user input into a validated state tuple.

    Example: "1,2,5,3,4,0,6,7,8" -> (1, 2, 5, 3, 4, 0, 6, 7, 8)

    Args:
        text: nine comma-separated digits.

    Returns:
        The state as a tuple of 9 ints.

    Raises:
        ValueError: if an entry is not an integer, or the digits 0-8 do not
            each appear exactly once.
    """
    # Split the text on commas and convert every piece to an int.
    state = text.split(',')
    final = ()
    for num in state:
        final = final + (int(num),)
    # Valid boards are permutations of 0..8 (this also rejects wrong lengths,
    # duplicates and out-of-range numbers).
    if sorted(final) != list(range(9)):
        raise ValueError("State must contain each digit from 0 to 8 exactly once")
    return final


def is_goal(state):
    """Return True if `state` is the goal configuration (shared goal test)."""
    return state == GOAL


def is_solvable(state):
    """
    Decide whether the goal can be reached from `state`.

    Exactly half of all arrangements are unsolvable. For a 3x3 board a state
    is solvable if and only if the number of inversions among the tiles
    (ignoring the blank) is even. An inversion is a pair of tiles where the
    larger one appears before the smaller one when the board is read row by row.

    Args:
        state: a state tuple.

    Returns:
        True if the goal is reachable from `state`, False otherwise.
    """
    myList = list(state)
    inversions = 0
    for i in range(len(myList)):
        if myList[i] == 0:      # the blank is not a tile; skip it
            continue
        for j in range(i+1,len(myList)):
            if myList[j] == 0:
                continue
            if myList[i] > myList[j]:
                inversions += 1
    return inversions%2==0


def get_neighbors(state):
    """
    Successor function: generate every state reachable in one move.

    The blank is swapped with the adjacent tile in each legal direction.
    Moves that would leave the 3x3 board are skipped.

    Args:
        state: a state tuple.

    Returns:
        A list of (move_name, new_state) pairs in the order of MOVES.
    """
    ret = []
    idx = state.index(0) #blank index
    row,col = divmod(idx,3) #blank row and column
    for name,row_ch,col_ch in MOVES:
        new_row = row + row_ch
        new_col = col + col_ch
        # Skip moves that fall outside the board.
        if new_row > 2 or new_col > 2 or new_row < 0 or new_col < 0:
            continue
        # Fresh copy per move so each child is built from the original state.
        myList = list(state)
        tile = myList[new_row*3+new_col]     # tile that slides into the blank
        myList[row*3+col] = tile
        myList[new_row*3+new_col] = 0
        ret.append((name,tuple(myList)))
    return ret


def reconstruct_path(parent, state):
    """
    Rebuild the solution path by following parent pointers back to the start.

    The searches store only a `parent` dictionary (child -> (parent, move))
    instead of a full path per node, which keeps memory low. The path is
    reconstructed once, when the goal is found.

    Args:
        parent: dict mapping each state to (previous_state, move), with the
            start state mapped to None.
        state: the state to trace back from (normally the goal).

    Returns:
        (moves, states), both ordered from the start state to `state`.
    """
    temp = [state]
    moves = []
    while parent[state]!=None:
        state, move = parent[state]
        temp.append(state)
        moves.append(move)
    # The walk went goal -> start, so reverse both lists.
    moves.reverse()
    temp.reverse()
    return moves, temp

def build_result(parent, goal_state, nodes_expanded, max_depth, frontier_history=None):
    """
    Package a finished search into the common result dictionary.

    A single builder guarantees that all five configurations report exactly
    the same fields (see "Result format" in the module docstring).
    """
    moves, states = reconstruct_path(parent, goal_state)
    
    frontiers_path = []
    if frontier_history:
        for state in states:
            # Get the snapshot for this state, default to empty string if not found
            snapshot = frontier_history.get(tuple(state), "")
            frontiers_path.append(snapshot)
    else:
        # Fallback if no history is provided
        frontiers_path = [""] * len(states)

    return {
        "moves": moves,
        "states": states,
        "cost": len(moves),
        "nodes_expanded": nodes_expanded,
        "max_depth": max_depth,
        "frontiers": frontiers_path 
    }
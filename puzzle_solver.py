"""
8-Puzzle Solver
===============
Artificial Intelligence course - Assignment 1

Solves the 8-puzzle with five search configurations and compares them:

    1. Breadth-First Search (BFS)
    2. Depth-First Search (DFS)
    3. Iterative-Deepening DFS (IDDFS)
    4. A* search with the Manhattan-distance heuristic
    5. A* search with the Euclidean-distance heuristic

Problem formulation
-------------------
* State   : a tuple of 9 integers read row by row; 0 is the blank.
            (1, 2, 5, 3, 4, 0, 6, 7, 8) represents the board

                1 2 5
                3 4 _
                6 7 8

            Tuples are immutable and hashable, so states can be stored in
            sets and used as dictionary keys.
* Goal    : (0, 1, 2, 3, 4, 5, 6, 7, 8), i.e. the blank in the top-left corner
            and the tiles in ascending order.
* Actions : move the blank Up, Down, Left or Right (each action costs 1).
            Children are always generated in the fixed order Up, Down, Left,
            Right so that every run is reproducible.

Result format
-------------
Every search function returns None when no solution is found, otherwise a dict:

    {
        "moves":          ["Up", "Left", "Left"],  # solution path as move names
        "states":         [start, ..., goal],      # every board along the path
        "cost":           3,                       # path cost = len(moves)
        "nodes_expanded": 10,                      # nodes expanded by the search
        "max_depth":      4,                       # deepest tree level reached
    }

run_algorithm() adds the running time ("time", in seconds) to this dict.

Assumptions and definitions (used identically by all algorithms)
----------------------------------------------------------------
* A node counts as "expanded" when it is removed from the frontier and is not
  the goal, i.e. when its children are generated. The goal node itself is not
  counted.
* "Cost" is the number of moves in the returned solution. "Max depth" is the
  deepest level of the search tree that the algorithm reached. They differ in
  general (for DFS they differ enormously).
* A solvable start state is required; the program checks solvability first
  (see is_solvable) so that no search runs forever on an impossible board.
"""

import heapq
import math
import time
from collections import deque

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Goal configuration: blank in the top-left corner, tiles in ascending order.
GOAL = (0, 1, 2, 3, 4, 5, 6, 7, 8)

# Possible moves of the blank as (move name, row change, column change).
# The list order is also the order in which children are expanded.
MOVES = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]


# =====================================================================
# 1. CORE HELPERS
#    Problem definition: input parsing, goal test, solvability test,
#    successor function and path reconstruction.
# =====================================================================

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


def build_result(parent, goal_state, nodes_expanded, max_depth):
    """
    Package a finished search into the common result dictionary.

    A single builder guarantees that all five configurations report exactly
    the same fields (see "Result format" in the module docstring).
    """
    moves, states = reconstruct_path(parent, goal_state)
    return {
        "moves": moves,
        "states": states,
        "cost": len(moves),
        "nodes_expanded": nodes_expanded,
        "max_depth": max_depth,
    }


# =====================================================================
# 2. HEURISTICS
#    Estimates h(n) of the remaining cost, used by A*.
#    Both ignore the blank. Both are admissible (they never overestimate
#    the true remaining cost), so A* returns optimal solutions.
#    Euclidean <= Manhattan for every state, so Manhattan is the more
#    informed heuristic and lets A* expand fewer nodes.
# =====================================================================

def manhattan(state):
    """
    Manhattan-distance heuristic.

    Sum over all tiles 1..8 of |row - goal_row| + |col - goal_col|, i.e. the
    number of grid steps each tile is away from its goal square.
    """
    total = 0
    for i in range(9):
        tile = state[i]
        if tile == 0:                       # skip the blank
            continue
        r, c = divmod(i, 3)                 # current row/column of the tile
        gr, gc = divmod(tile, 3)            # goal row/column (tile t belongs at index t)
        total += abs(r - gr) + abs(c - gc)
    return total


def euclidean(state):
    """
    Euclidean-distance heuristic.

    Sum over all tiles 1..8 of the straight-line distance
    sqrt(row_difference^2 + col_difference^2) to the tile's goal square.
    """
    total = 0
    for i in range(9):
        tile = state[i]
        if tile == 0:                       # skip the blank
            continue
        r, c = divmod(i, 3)                 # current row/column of the tile
        gr, gc = divmod(tile, 3)            # goal row/column
        total += math.sqrt((r - gr) ** 2 + (c - gc) ** 2)
    return total


# =====================================================================
# 3. SEARCH ALGORITHMS
#    All five configurations use the same counting rules so that the
#    comparison in the report is fair:
#      * nodes_expanded is incremented when a non-goal node is removed
#        from the frontier and its children are generated;
#      * max_depth is the deepest tree level reached;
#      * cost is len(moves) of the solution found.
# =====================================================================

def bfs(start):
    """
    Breadth-first search.

    Expands nodes level by level using a FIFO queue, so the first time the
    goal is reached it is reached by a shortest path (optimal for unit costs).
    Drawback: the frontier can grow very large.

    Data structures:
        frontier - collections.deque used as a FIFO queue
        parent   - dict child -> (parent, move); doubles as the "seen" set
        depth    - dict state -> depth in the search tree

    Args:
        start: the initial state tuple.

    Returns:
        The result dict, or None if the goal is unreachable.
    """
    frontier = deque([start])
    parent = {start:None}
    depth = {start: 0}
    nodes_expanded = 0
    max_depth = 0
    while frontier:
        state = frontier.popleft()
        if is_goal(state):
            return build_result(parent,state,nodes_expanded,max_depth)
        nodes_expanded+=1
        for move,new_state in get_neighbors(state):
            # Only unseen states are added, so no state is queued twice.
            if new_state not in parent:
                parent[new_state] = (state, move)
                depth[new_state] = depth[state] + 1
                max_depth = max(max_depth,depth[new_state])
                frontier.append(new_state)
    return None


def dfs(start):
    """
    Depth-first search (iterative).

    Follows one branch as deep as possible before backtracking, using a stack.
    It needs little memory but does NOT guarantee a shortest path; the
    returned path is typically very long. It is iterative on purpose:
    recursion would exceed Python's recursion limit on paths this deep.

    A state's parent and depth are recorded when the state is popped (not when
    it is pushed). Each stack entry therefore carries the state that pushed it,
    so the reported path is exactly the path the search walked.

    Data structures:
        frontier - Python list used as a LIFO stack of (state, pusher, move)
        parent   - dict child -> (parent, move); doubles as the "visited" set
        depth    - dict state -> depth in the search tree

    Args:
        start: the initial state tuple.

    Returns:
        The result dict, or None if the goal is unreachable.
    """
    frontier = [(start, None, None)]            # entries: (state, pusher, move)
    parent = {}                                 # filled when a state is popped
    depth = {}                                  # filled when a state is popped
    nodes_expanded = 0
    max_depth = 0
    while frontier:
        state, par, move = frontier.pop()       # most recently pushed entry
        if state in parent:                     # duplicate entry: already visited
            continue
        if par is None:                         # the start state
            parent[state] = None
            depth[state] = 0
        else:                                   # record the parent at pop time
            parent[state] = (par, move)
            depth[state] = depth[par] + 1
        max_depth = max(max_depth, depth[state])
        if is_goal(state):
            return build_result(parent, state, nodes_expanded, max_depth)
        nodes_expanded += 1
        # Push children in reverse so that Up is popped (explored) first.
        for move, new_state in reversed(get_neighbors(state)):
            if new_state not in parent:         # skip already visited states
                frontier.append((new_state, state, move))
    return None


def depth_limited_search(start, limit):
    """
    Depth-first search that does not expand nodes deeper than `limit`.

    Building block of iterative deepening. A plain "visited" set cannot be
    used here: a state first reached via a long path would block a shorter
    path to the same state and the optimal solution could be missed. Instead,
    `best_depth` stores the shallowest depth at which each state has been
    reached, and a state is skipped only if it was already reached at the same
    or a shallower depth.

    Args:
        start: the initial state tuple.
        limit: maximum depth at which nodes are still expanded.

    Returns:
        (result_or_None, nodes_expanded, max_depth) for this single pass.
    """
    frontier = [(start, None, None, 0)]         # entries: (state, pusher, move, depth)
    parent = {}
    best_depth = {}                             # shallowest depth reached per state
    nodes_expanded = 0
    max_depth = 0
    while frontier:
        state, par, move, d = frontier.pop()
        # Skip if this state was already reached at the same or smaller depth.
        if state in best_depth and best_depth[state] <= d:
            continue
        best_depth[state] = d
        parent[state] = None if par is None else (par, move)
        max_depth = max(max_depth, d)
        if is_goal(state):
            return build_result(parent, state, nodes_expanded, max_depth), nodes_expanded, max_depth
        nodes_expanded += 1
        if d == limit:                          # depth limit reached: do not expand further
            continue
        for mv, new_state in reversed(get_neighbors(state)):
            frontier.append((new_state, state, mv, d + 1))
    return None, nodes_expanded, max_depth


def iddfs(start):
    """
    Iterative-deepening depth-first search.

    Runs depth_limited_search with limit = 0, 1, 2, ... and stops at the first
    limit that finds the goal. That gives the shortest path (like BFS) with
    the small memory footprint of DFS, at the price of re-expanding the upper
    levels on every pass. The reported nodes_expanded is the total over ALL
    passes, which makes this repeated work visible in the comparison.

    The loop terminates because the program only runs on solvable states.

    Args:
        start: the initial state tuple (must be solvable).

    Returns:
        The result dict.
    """
    total_nodes = 0
    limit = 0
    while True:
        result, nodes, max_d = depth_limited_search(start, limit)
        total_nodes += nodes                    # accumulate work from every pass
        if result is not None:
            result["nodes_expanded"] = total_nodes
            return result
        limit += 1


def astar(start, heuristic):
    """
    A* search.

    Always expands the frontier node with the smallest f(n) = g(n) + h(n),
    where g is the number of moves made so far and h is the heuristic estimate
    of the moves remaining. With an admissible heuristic the first goal that is
    removed from the frontier is reached by an optimal path.

    Data structures:
        frontier - heapq priority queue of (f, tie_breaker, state)
        g_cost   - dict state -> cheapest known cost from the start
        parent   - dict child -> (parent, move)
        closed   - set of states that have already been expanded

    Args:
        start: the initial state tuple.
        heuristic: function state -> estimate (manhattan or euclidean).

    Returns:
        The result dict, or None if the goal is unreachable.
    """
    counter = 0                                 # tie-breaker: states are never compared
    frontier = [(heuristic(start), counter, start)]
    parent = {start: None}
    g_cost = {start: 0}
    closed = set()
    nodes_expanded = 0
    max_depth = 0
    while frontier:
        f, _, state = heapq.heappop(frontier)
        if state in closed:                     # stale entry: a cheaper path was expanded first
            continue
        if is_goal(state):                      # goal test on removal guarantees optimality
            return build_result(parent, state, nodes_expanded, max_depth)
        closed.add(state)
        nodes_expanded += 1
        for move, new_state in get_neighbors(state):
            new_g = g_cost[state] + 1           # every move costs 1
            # Record the neighbour if it is new or reached by a cheaper path.
            if new_state not in g_cost or new_g < g_cost[new_state]:
                g_cost[new_state] = new_g
                parent[new_state] = (state, move)
                max_depth = max(max_depth, new_g)
                counter += 1
                heapq.heappush(frontier, (new_g + heuristic(new_state), counter, new_state))
    return None


# =====================================================================
# 4. OUTPUT AND REPORTING
#    Board drawing, step-by-step solution trace, timing wrapper and the
#    summary table used for the comparison.
# =====================================================================

def format_board(state):
    """
    Render a state as a 3-line text board with the blank shown as '_'.

    Shared by every printing function so all output looks the same.
    """
    board = ""
    myList = list(state)
    for row in range(3):
        for col in range(3):
            val = myList[row*3+col]
            if val == 0:
                board += "_"
            else:
                board += str(val)
            board += " "
        board += "\n"
    return board


def print_solution(result, start, show_steps=True, max_steps=50):
    """
    Print the solution in a traceable step-by-step format.

    Shows the start board, then each move together with the resulting board,
    and finally a summary line (cost, nodes expanded, max depth). Paths longer
    than `max_steps` (e.g. DFS paths with tens of thousands of moves) print
    only their first 10 and last 10 steps; the full path is saved to a file by
    run_algorithm.

    Args:
        result: result dict returned by a search function.
        start: the initial state (printed first).
        show_steps: if False, print only the summary line.
        max_steps: longest path that is printed in full.
    """
    moves = result["moves"]
    states = result["states"]
    total = len(moves)

    def show_step(i):                                   # prints step number i+1
        print(f"Step {i + 1}: {moves[i]}")
        print(format_board(states[i + 1]), end="")

    print("Start:")
    print(format_board(start), end="")
    if show_steps:
        if total <= max_steps:
            for i in range(total):
                show_step(i)
        else:                                           # long path (DFS): first 10 and last 10 only
            for i in range(10):
                show_step(i)
            print(f"... {total - 20} steps omitted ...")
            for i in range(total - 10, total):
                show_step(i)
    print(f"Cost: {result['cost']} | Nodes expanded: {result['nodes_expanded']} | "
          f"Max depth: {result['max_depth']}")


def write_trace(label, result, start, filename):
    """Write the complete step-by-step solution (no omissions) to a text file."""
    with open(filename, "w") as f:
        f.write(f"{label}\nStart:\n{format_board(start)}")
        for i, move in enumerate(result["moves"]):
            f.write(f"Step {i + 1}: {move}\n{format_board(result['states'][i + 1])}")


def run_algorithm(label, fn, start):
    """
    Run one search configuration with timing and print its trace.

    Args:
        label: display name, e.g. "A* Manhattan".
        fn: function taking the start state and returning a result dict or
            None (a lambda is used to bind the heuristic for A*).
        start: the initial state tuple.

    Returns:
        The result dict with an added "time" entry (seconds), or None if no
        solution was found.
    """
    print("=" * 50)
    print(label)
    print("=" * 50)
    t0 = time.perf_counter()
    result = fn(start)
    elapsed = time.perf_counter() - t0
    if result is None:
        print("No solution found.")
        return None
    result["time"] = elapsed
    print_solution(result, start)
    print(f"Time: {elapsed:.4f} s")
    if result["cost"] > 50:                             # save the full trace for long paths
        filename = label.replace(" ", "_").replace("*", "star") + "_trace.txt"
        write_trace(label, result, start, filename)
        print(f"(full {result['cost']}-step trace saved to {filename})")
    return result


def print_summary_table(results):
    """
    Print one comparison row per algorithm.

    Args:
        results: list of (label, result_dict) pairs.
    """
    print()
    print(f"{'Algorithm':<20}{'Cost':>8}{'Expanded':>12}{'Max depth':>11}{'Time (s)':>10}")
    print("-" * 61)
    for label, r in results:
        print(f"{label:<20}{r['cost']:>8}{r['nodes_expanded']:>12}{r['max_depth']:>11}{r['time']:>10.3f}")


# =====================================================================
# 5. TESTS AND MAIN PROGRAM
# =====================================================================

def run_tests():
    """Quick sanity checks of the core helpers; stops the program if one fails."""
    assert is_solvable((1,2,5,3,4,0,6,7,8))
    assert is_solvable(GOAL)
    assert not is_solvable((0,2,1,3,4,5,6,7,8))      # two tiles swapped: unsolvable
    assert bfs((1,2,5,3,4,0,6,7,8))["moves"] == ["Up", "Left", "Left"]
    assert manhattan(GOAL) == 0
    print("All tests passed.")


def main():
    """
    Program entry point.

    Reads the initial state, validates it, checks that it is solvable, runs
    all five search configurations on it and prints a comparison table.
    """
    text = input("Enter the initial state (e.g. 1,2,5,3,4,0,6,7,8): ")
    try:
        start = parse_state(text)
    except ValueError as e:
        print("Invalid state:", e)
        return
    # Without this check, uninformed searches would exhaust the whole
    # reachable space on an unsolvable board (and IDDFS would never stop).
    if not is_solvable(start):
        print("This state is unsolvable (odd number of inversions). No search was run.")
        return

    # (display name, function that takes only the start state)
    configs = [
        ("BFS", bfs),
        ("DFS", dfs),
        ("IDDFS", iddfs),
        ("A* Manhattan", lambda s: astar(s, manhattan)),
        ("A* Euclidean", lambda s: astar(s, euclidean)),
    ]
    results = []
    for label, fn in configs:
        r = run_algorithm(label, fn, start)
        if r is not None:
            results.append((label, r))
    print_summary_table(results)


if __name__ == "__main__":
    run_tests()
    main()
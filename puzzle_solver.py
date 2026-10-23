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
from collections import deque

from utils.Constants import *
from utils.core_helpers import *
from utils.output_reporting import *

# =====================================================================
# 1. HEURISTICS
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
# 2. SEARCH ALGORITHMS
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
    
    frontier_history = {}  # NEW: Track Queue snapshots

    while frontier:
        # Snapshot the Queue before popping
        snapshot = f"Q-Size: {len(frontier)} | Top: {frontier[0]}"
        
        state = frontier.popleft()
        frontier_history[tuple(state)] = snapshot  # Save snapshot for this state
        
        if is_goal(state):
            return build_result(parent, state, nodes_expanded, max_depth, frontier_history)
        
        nodes_expanded += 1
        for move, new_state in get_neighbors(state):
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

    frontier_history = {}  # Track Stack snapshots

    while frontier:
        # Snapshot the Stack before popping (Top is at the end of the list: index -1)
        snapshot = f"Stack-Size: {len(frontier)} | Top: {frontier[-1][0]}"
        
        state, par, move = frontier.pop()       
        frontier_history[tuple(state)] = snapshot  # Save snapshot
        
        if state in parent:                     
            continue
        if par is None:                         # the start state
            parent[state] = None
            depth[state] = 0
        else:                                   # record the parent at pop time
            parent[state] = (par, move)
            depth[state] = depth[par] + 1
        max_depth = max(max_depth, depth[state])
        if is_goal(state):
            return build_result(parent, state, nodes_expanded, max_depth, frontier_history)
            
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
    
    frontier_history = {}  # Track Stack snapshots

    while frontier:
        # Snapshot Stack
        snapshot = f"Stack-Size: {len(frontier)} | Top: {frontier[-1][0]}"
        
        state, par, move, d = frontier.pop()
        frontier_history[tuple(state)] = snapshot  # Save snapshot
        
        if state in best_depth and best_depth[state] <= d:
            continue
        best_depth[state] = d
        parent[state] = None if par is None else (par, move)
        max_depth = max(max_depth, d)
        if is_goal(state):
            # Pass frontier_history into build_result
            return build_result(parent, state, nodes_expanded, max_depth, frontier_history), nodes_expanded, max_depth
            
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
    
    frontier_history = {}  # Track Heap snapshots

    while frontier:
        # Snapshot the Heap (Top is at index 0, format: f(n))
        top_f_score = frontier[0][0]
        snapshot = f"Heap-Size: {len(frontier)} | Top f(n): {top_f_score}"
        
        f, _, state = heapq.heappop(frontier)
        if state in closed:                     # stale entry: a cheaper path was expanded first
            continue
        
        frontier_history[tuple(state)] = snapshot
            
        if is_goal(state):                      
            return build_result(parent, state, nodes_expanded, max_depth, frontier_history)
            
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
# 3. TESTS AND MAIN PROGRAM
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
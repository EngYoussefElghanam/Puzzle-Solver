# =====================================================================
#    OUTPUT AND REPORTING
#    Board drawing, step-by-step solution trace, timing wrapper and the
#    summary table used for the comparison.
# =====================================================================

import time

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
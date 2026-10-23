import tkinter as tk
from tkinter import ttk, messagebox
import threading

from utils.core_helpers import *
from puzzle_solver import *

class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.geometry("850x520")  

        # --- State Variables ---
        self.entries = []   # The 9 boxes
        self.states = []    # The sequence of board states
        self.moves = []     # The sequence of moves taken to reach the goal state
        self.frontiers = [] # Snapshots of the frontier for each step
        self.is_animating = False
        self.skip_animation = False 
        self.setup_ui()

    def setup_ui(self):
        """
        This method builds the visual layout of the application.
        Tkinter uses "geometry managers" like .pack() and .grid() to place items.
        """
        
        # --- Main Layout Frames ---

        # Create a frame for the left half of the screen (holds the board and buttons)
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT)

        # Create a frame for the right half of the screen (holds the step-by-step table)
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- LEFT PANEL: Board Grid ---
        board_frame = tk.Frame(left_frame)
        board_frame.pack() # Default packing puts it at the top of the left_frame

        default_start = [1, 2, 5, 3, 4, 0, 8, 6, 7]

        for i in range(9):
            row, col = divmod(i, 3)
            
            # create 9 text boxes which the user can type into
            entry = tk.Entry(board_frame, width=2, font=("Helvetica", 36, "bold"), justify="center")
            entry.grid(row=row, column=col, padx=5, pady=5)
            val = default_start[i]
            entry.insert(0, str(val) if val != 0 else "0")
            self.entries.append(entry)

        # --- LEFT PANEL: Controls ---
        # Frame for the dropdown and buttons
        controls_frame = tk.Frame(left_frame)
        controls_frame.pack(pady=20) # 20 pixels of space above and below it

        # Plain text 
        tk.Label(controls_frame, text="Method:").grid(row=0, column=0, padx=5, sticky="e")
        
        # Dropdown menu
        self.algo_combo = ttk.Combobox(controls_frame, values=["BFS", "DFS", "IDDFS", "A* Manhattan", "A* Euclidean"], state="readonly")
        # Set the default to BFS
        self.algo_combo.current(0)
        self.algo_combo.grid(row=0, column=1, padx=5)

        # Start button 
        self.start_btn = tk.Button(
            controls_frame, text="Start Solving", command=self.start_solver, 
            bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold") 
        )
        self.start_btn.grid(row=1, column=0, columnspan=2, pady=(15, 5), padx=5, sticky="we")

        # Skip button (disabled)
        self.skip_btn = tk.Button(
            controls_frame, text="Skip", command=self.skip_to_end, 
            bg="#FF9800", fg="white", font=("Helvetica", 12, "bold"), state=tk.DISABLED
        )
        self.skip_btn.grid(row=2, column=0, columnspan=2, pady=(0, 15), padx=5, sticky="we")

        # --- RIGHT PANEL: Data Table ---
        
        # Create the Table
        columns = ("step", "move", "frontier")
        self.trace_tree = ttk.Treeview(right_frame, columns=columns, show="headings", height=20)

        self.trace_tree.heading("step", text="Step")
        self.trace_tree.column("step", width=60, anchor=tk.CENTER)        
        self.trace_tree.heading("move", text="Move")
        self.trace_tree.column("move", width=60, anchor=tk.CENTER)
        self.trace_tree.heading("frontier", text="Frontier Snapshot")
        self.trace_tree.column("frontier", width=300, anchor=tk.W) 

        self.trace_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for the table
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.trace_tree.yview)
        self.trace_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def start_solver(self):
        if self.is_animating:
            return

        # 1. Read the user's input from the 9 text boxes
        vals = []
        for e in self.entries:
            val = e.get().strip()
            if val == "": val = "0" 
            vals.append(val)
            
        # Join the list into a comma-separated string (e.g., "1,2,5,3,4,0,8,6,7")
        text_input = ",".join(vals)

        # 2. Parse and Validate the input using your backend function
        try:
            state = parse_state(text_input)
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error: {e}")
            return

        # 3. Check if the puzzle is mathematically solvable (checking for odd inversions)
        if not is_solvable(state):
            messagebox.showerror("Unsolvable", "This puzzle configuration is not solvable (odd inversions).")
            return

        
        # Reset the GUI for a new run
        self.start_btn.config(state=tk.DISABLED)
        self.skip_btn.config(state=tk.DISABLED)
        self.skip_animation = False
        for item in self.trace_tree.get_children():
            self.trace_tree.delete(item)
            
        # Set Algorithm 
        algo_name = self.algo_combo.get()
        self.trace_tree.insert("", tk.END, values=("-", "-", f"Running {algo_name}..."))
        self.root.update()

        # 4. Run the search in a BACKGROUND THREAD
        threading.Thread(target=self.run_search_thread, args=(algo_name, state), daemon=True).start()

    def run_search_thread(self, algo_name, start_state):
        result = None
        
        if algo_name == "BFS":
            result = bfs(start_state)
        elif algo_name == "DFS":
            result = dfs(start_state)
        elif algo_name == "IDDFS":
            result = iddfs(start_state)
        elif algo_name == "A* Manhattan":
            result = astar(start_state, manhattan)
        elif algo_name == "A* Euclidean":
            result = astar(start_state, euclidean)

        # pass the result back to the main GUI thread for processing and animation
        self.root.after(0, lambda: self.process_result(result))

    def process_result(self, result):
        # Remove the temporary "Running..." row
        for item in self.trace_tree.get_children():
            self.trace_tree.delete(item)

        # If the solver returned None, it couldn't find a path
        if result is None:
            self.trace_tree.insert("", tk.END, values=("Error", "-", "Search failed or not hooked up."))
            self.start_btn.config(state=tk.NORMAL) # Re-enable the start button
            return

        self.states = result["states"]
        self.moves = result["moves"]
        self.frontiers = result.get("frontiers", []) 

        # Insert three rows at the top of the table showing the metadata (Cost and Nodes Expanded)
        self.trace_tree.insert("", tk.END, values=("Cost", result['cost'], ""))
        self.trace_tree.insert("", tk.END, values=("Expanded", result['nodes_expanded'], ""))
        self.trace_tree.insert("", tk.END, values=("---", "---", "---"))

        self.is_animating = True
        self.skip_animation = False

        # Disable Boxes
        for e in self.entries:
            e.config(state="disabled")

        # Enable the Skip button 
        self.skip_btn.config(state=tk.NORMAL) 
        
        # Start the animation
        self.animate_step(0)

    def skip_to_end(self):
        if self.is_animating:
            self.skip_animation = True

    def animate_step(self, step_idx):
        """
        This is a recursive-like loop, but instead of calling itself immediately,
        it uses Tkinter's root.after() to schedule the next step in the future.
        """
        
        # --- 1. Handle Skip Request ---
        if self.skip_animation:
            # Grab the very last state from our solution list
            final_state = self.states[-1]
            
            # Update the physical text boxes to show the final state
            for e in self.entries:
                e.config(state="normal", bg="white", fg="black") # reset colors
            for i, val in enumerate(final_state):
                self.entries[i].delete(0, tk.END)
                self.entries[i].insert(0, str(val) if val != 0 else "")

            # Run a fast loop to dump all remaining un-animated steps into the Treeview table at once
            for idx in range(step_idx, len(self.states)):
                if idx == 0:
                    frontier_str = self.frontiers[0] if len(self.frontiers) > 0 else ""
                    self.trace_tree.insert("", tk.END, values=("Start", "None", frontier_str))
                else:
                    move_str = self.moves[idx - 1]
                    frontier_str = self.frontiers[idx] if idx < len(self.frontiers) else ""
                    self.trace_tree.insert("", tk.END, values=(f"{idx}", move_str, frontier_str))

            self.is_animating = False
            self.start_btn.config(state=tk.NORMAL)
            self.skip_btn.config(state=tk.DISABLED)

            item_id = self.trace_tree.insert("", tk.END, values=("Goal", "Reached!", ""))
            # make the scrollbar scroll down to the newest item
            self.trace_tree.see(item_id)
            
            return

        # --- 2. Normal Completion ---
        # If we have animated all the steps...
        if step_idx >= len(self.states):
            self.is_animating = False
            self.start_btn.config(state=tk.NORMAL)
            self.skip_btn.config(state=tk.DISABLED)
            
            # Reset colors on all text boxes to normal
            for e in self.entries:
                e.config(state="normal", bg="white", fg="black")
                
            item_id = self.trace_tree.insert("", tk.END, values=("Goal", "Reached!", ""))
            # make the scrollbar scroll down to the newest item
            self.trace_tree.see(item_id)
            
            return

        # --- 3. Normal Animation Step ---
        current_state = self.states[step_idx]

        # Briefly enable the text boxes so we can change their text and reset their colors
        for e in self.entries:
            e.config(state="normal", bg="white", fg="black") 

        # Update the 9 text boxes with the numbers for this current step
        for i, val in enumerate(current_state):
            self.entries[i].delete(0, tk.END)
            self.entries[i].insert(0, str(val) if val != 0 else "")

        # highlight the tile that just moved
        if step_idx > 0:
            prev_state = self.states[step_idx - 1]
            moving_tile_idx = prev_state.index(0)
            self.entries[moving_tile_idx].config(bg="#FFD700", fg="#B22222")

            # Grab the text data for the Treeview
            move_str = self.moves[step_idx - 1]
            frontier_str = self.frontiers[step_idx] if step_idx < len(self.frontiers) else "" # Prevent list index errors

            # Add the row to the table
            item_id = self.trace_tree.insert("", tk.END, values=(f"{step_idx}", move_str, frontier_str))
            self.trace_tree.see(item_id) 
            
        elif step_idx == 0:
            # Special case for step 0 (the initial board layout). No move has happened yet.
            frontier_str = self.frontiers[0] if len(self.frontiers) > 0 else ""
            item_id = self.trace_tree.insert("", tk.END, values=("Start", "None", frontier_str))
            self.trace_tree.see(item_id)

        # Set all text boxes to 'readonly' so they look like flat text but block the user from typing
        for e in self.entries:
            e.config(state="readonly")

        # Schedule the next animation step.
        self.root.after(500, self.animate_step, step_idx + 1)

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()
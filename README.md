# 🎮 Multiplayer Game – Pathfinding AI

**SDG 9: Industry, Innovation & Infrastructure**

An NPC (non-player character) finds the optimal path to a player on a map full of obstacles. When it cannot see any player, it wanders around using depth-first search.

🔗 **Live demo:** [https://multiplayers-game-kxrc95y4jkyudh2jc6bygy.streamlit.app/](
multiplayers-game-kxrc95y4jkyudh2jc6bygy
.streamlit.app)

## Features

- **A\* search** with a Euclidean-distance heuristic (also selectable: octile, Manhattan, zero/Dijkstra)
- **Min-heap** (hand-written binary heap) as the A* open set to get the lowest f-score node quickly
- **DFS wander** behaviour when the player is not visible
- **Vision system:** vision radius plus line of sight (Bresenham)
- **Multiplayer:** 1 to 4 players, and the NPC chases the nearest visible one
- **Map types:** random obstacle maps and DFS-generated mazes
- **Live simulation** with play and step-through controls

## Tech stack

| Part | Technology |
|---|---|
| Backend (AI) | C++17 |
| Frontend | Python, Streamlit, Matplotlib |

## Project structure

```
├── app.py               # Streamlit frontend
├── requirements.txt     # Python dependencies
├── packages.txt         # g++ compiler for Streamlit Cloud
└── backend/
    ├── pathfinding.hpp  # MinHeap, A*, DFS wander, line of sight, simulation
    ├── main.cpp         # command-line backend (outputs JSON)
    └── Makefile
```

## How it works

1. `app.py` compiles the C++ backend the first time it starts.
2. The frontend sends the map to the backend, which returns the result as JSON.
3. Streamlit draws the map, paths and simulation.

### A* algorithm
Each cell has `g` (cost from the NPC) and `h` (estimated cost to the player). A* always expands the open node with the smallest `f = g + h`. Straight moves cost 1 and diagonal moves cost √2, and diagonals cannot cut corners around walls. Euclidean distance never overestimates the real cost, so the path found is optimal.

### Min-heap
A binary heap stored in a `vector`. Push and pop take O(log n), and reading the best node takes O(1). Overall A* runs in O(E log V).

### DFS wander
The NPC moves to a random unvisited neighbour, backtracks at dead ends, and restarts when the whole reachable area has been explored.

### NPC behaviour
```
WANDER ──(player in vision + line of sight)──▶ CHASE
CHASE  ──(player lost)──────────────────────▶ WANDER
```

## Run locally

Requirements: Python 3.9+ and a C++ compiler (g++ or clang++).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

Test the backend on its own:

```bash
cd backend
g++ -O2 -std=c++17 main.cpp -o npc_ai
./npc_ai test
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io]() and sign in with GitHub.
3. Click **Create app**, select this repo, and set the main file to `app.py`.
4. Click **Deploy**.

## SDG 9 connection

Efficient path-planning is a building block for innovation in games, robotics and logistics. It helps machines move through complex environments using less computation and energy.

## Author
VEERA PURNIMA VENKATA

# Multiplayer Game – Pathfinding AI (SDG 9)

An NPC chases players on a map full of obstacles.

| Requirement | Where |
|---|---|
| A* with Euclidean heuristic | `aStar()` in `backend/pathfinding.hpp` |
| Min-heap open set | `MinHeap` in `backend/pathfinding.hpp` |
| DFS "wander" when the player is not visible | `DFSWanderer` in `backend/pathfinding.hpp` |
| Visibility (vision radius + line of sight) | `hasLineOfSight()` / `canSee()` |
| Frontend | `app.py` (Streamlit) |

```
npc-pathfinding/
├── app.py              # Streamlit frontend (compiles + calls the C++ backend)
├── requirements.txt    # streamlit, matplotlib, numpy
├── packages.txt        # g++  (needed on Streamlit Community Cloud)
└── backend/
    ├── pathfinding.hpp # MinHeap, A*, DFS wander, line of sight, simulation
    ├── main.cpp        # CLI: astar | wander | simulate | test  (JSON output)
    └── Makefile
```

## Run it on your computer

Requirements: Python 3.9+ and a C++ compiler (g++ or clang++).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens at http://localhost:8501. The app compiles the C++ backend automatically the first time.

Optional – build and test the backend on its own:

```bash
cd backend
make          # or: g++ -O2 -std=c++17 main.cpp -o npc_ai
./npc_ai test # prints ALL TESTS PASSED
printf '.....\n.###.\n.....\n' | ./npc_ai astar --npc 0,0 --player 4,2
```

## Get a public Streamlit URL (free)

1. Create a GitHub repository and upload this whole folder (keep the structure; `packages.txt` and `requirements.txt` must be in the root next to `app.py`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repository and branch → **Main file path:** `app.py`.
4. Click **Deploy**. After a few minutes you get a link like `https://your-app-name.streamlit.app` that you can share.

`packages.txt` installs `g++` on the server, and `app.py` compiles the backend on first start, so there is nothing else to configure.

Quick alternative (no GitHub): run locally and share a temporary tunnel, e.g. `ngrok http 8501` or `cloudflared tunnel --url http://localhost:8501`.

## How the pieces talk

`app.py` writes the map (one row per line, `#` = wall) to the backend's stdin and reads back JSON:

```
npc_ai astar    --npc x,y --player x,y [--diag 1] [--heur euclidean|octile|manhattan|zero] [--explored 1]
npc_ai wander   --npc x,y [--steps 60] [--seed 1]
npc_ai simulate --npc x,y --players x,y:x,y [--vision 8] [--ticks 150] [--seed 1] [--period 2]
```

## Algorithms in short

* **A\***: expands the open node with the smallest `f = g + h`. Straight moves cost 1, diagonals √2 (no corner cutting). Euclidean distance never overestimates, so the path is optimal.
* **Min-heap**: binary heap in a `vector`; push/pop O(log n). Improved nodes are pushed again and stale entries are skipped (lazy deletion). Total cost O(E log V).
* **DFS wander**: moves to a random unvisited neighbour, backtracks at dead ends, restarts when the reachable area is fully explored.
* **NPC state machine**: `WANDER` → (player within vision radius and in line of sight) → `CHASE` → (player lost) → `WANDER`. With several players the NPC chases the nearest visible one.

## Ideas to extend

Weighted terrain (mud/water), several NPCs, jump-point search, a "last known position" search state, user-drawn maps.

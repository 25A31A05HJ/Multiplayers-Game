"""Streamlit frontend for the NPC Pathfinding AI project.

The AI itself (A*, min-heap, DFS wander, simulation) is written in C++ in
backend/.  This app compiles that backend automatically (g++ / clang++), runs
it as a subprocess and visualises the JSON it returns.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import json
import platform
import random
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent


def _find_backend_dir() -> Path:
    """Folder that holds main.cpp + pathfinding.hpp (backend/ or next to app.py)."""
    for d in (ROOT / "backend", ROOT):
        if (d / "main.cpp").is_file() and (d / "pathfinding.hpp").is_file():
            return d
    return ROOT / "backend"


BACKEND_DIR = _find_backend_dir()
EXE = BACKEND_DIR / ("npc_ai.exe" if platform.system() == "Windows" else "npc_ai")

PLAYER_COLORS = ["#1d7be0", "#2a9d8f", "#f4a261", "#9b5de5", "#06d6a0", "#ef476f", "#118ab2", "#8d6e63"]
NPC_COLOR = "#e63946"


# ---------------------------------------------------------------- backend
@st.cache_resource(show_spinner="Compiling the C++ backend...")
def build_backend() -> str:
    """Compile backend/main.cpp (only if the binary is missing or outdated)."""
    sources = [BACKEND_DIR / "main.cpp", BACKEND_DIR / "pathfinding.hpp"]
    missing = [s.name for s in sources if not s.is_file()]
    if missing:
        found = sorted(p.name for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts)
        raise RuntimeError(
            f"C++ source file(s) not found: {', '.join(missing)}. Put main.cpp and pathfinding.hpp in a "
            f"folder named 'backend' next to app.py (or next to app.py itself) and push again.\n"
            f"Files the app can see: {', '.join(found)}"
        )
    newest_source = max(s.stat().st_mtime for s in sources)
    if EXE.exists() and EXE.stat().st_mtime >= newest_source:
        return str(EXE)
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError(
            "No C++ compiler found. Install g++ (Linux: sudo apt install g++, "
            "macOS: xcode-select --install, Windows: MinGW-w64) and reload. "
            "On Streamlit Cloud add 'g++' to packages.txt."
        )
    proc = subprocess.run(
        [compiler, "-O2", "-std=c++17", str(BACKEND_DIR / "main.cpp"), "-o", str(EXE)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("C++ compilation failed:\n" + proc.stderr)
    return str(EXE)


def call_backend(exe: str, command: str, grid_rows: list[str], **params) -> dict:
    args = [exe, command]
    for key, value in params.items():
        args += [f"--{key}", str(value)]
    proc = subprocess.run(
        args, input="\n".join(grid_rows) + "\n", capture_output=True, text=True, timeout=60
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Backend returned invalid output: {proc.stderr or proc.stdout}")
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def pt(p) -> str:
    return f"{p[0]},{p[1]}"


# ------------------------------------------------------------ map building
def random_map(w: int, h: int, density: float, rng: random.Random) -> list[list[str]]:
    grid = [["#" if rng.random() < density else "." for _ in range(w)] for _ in range(h)]
    # a few long walls with a gap, so the map has rooms and corridors
    for _ in range(max(2, (w + h) // 12)):
        if rng.random() < 0.5:
            y, x0 = rng.randrange(2, h - 2), rng.randrange(1, w // 2)
            length = rng.randrange(4, max(5, w // 2))
            gap = rng.randrange(length)
            for i in range(length):
                if x0 + i < w - 1 and i != gap:
                    grid[y][x0 + i] = "#"
        else:
            x, y0 = rng.randrange(2, w - 2), rng.randrange(1, h // 2)
            length = rng.randrange(4, max(5, h // 2))
            gap = rng.randrange(length)
            for i in range(length):
                if y0 + i < h - 1 and i != gap:
                    grid[y0 + i][x] = "#"
    for x in range(w):
        grid[0][x] = grid[h - 1][x] = "#"
    for y in range(h):
        grid[y][0] = grid[y][w - 1] = "#"
    return grid


def maze_map(w: int, h: int, rng: random.Random) -> list[list[str]]:
    """Maze carved with an iterative depth-first search (recursive backtracker)."""
    w |= 1
    h |= 1
    grid = [["#"] * w for _ in range(h)]
    grid[1][1] = "."
    stack = [(1, 1)]
    while stack:
        x, y = stack[-1]
        options = [
            (x + dx, y + dy, dx, dy)
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
            if 0 < x + dx < w - 1 and 0 < y + dy < h - 1 and grid[y + dy][x + dx] == "#"
        ]
        if options:
            nx, ny, dx, dy = rng.choice(options)
            grid[y + dy // 2][x + dx // 2] = "."
            grid[ny][nx] = "."
            stack.append((nx, ny))
        else:
            stack.pop()
    for _ in range((w * h) // 25):  # knock out some walls -> loops / shortcuts
        grid[rng.randrange(1, h - 1)][rng.randrange(1, w - 1)] = "."
    return grid


def largest_component(rows: list[str]) -> list[tuple[int, int]]:
    h, w = len(rows), len(rows[0])
    seen = [[False] * w for _ in range(h)]
    best: list[tuple[int, int]] = []
    for sy in range(h):
        for sx in range(w):
            if rows[sy][sx] == "#" or seen[sy][sx]:
                continue
            comp, queue = [], deque([(sx, sy)])
            seen[sy][sx] = True
            while queue:
                x, y = queue.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and rows[ny][nx] != "#" and not seen[ny][nx]:
                        seen[ny][nx] = True
                        queue.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    return best


@st.cache_data(show_spinner=False)
def build_scene(map_type: str, width: int, height: int, density: float, seed: int, n_players: int):
    rng = random.Random(seed)
    if map_type.startswith("Maze"):
        grid = maze_map(width, height, rng)
    else:
        grid = random_map(width, height, density, rng)
    rows = ["".join(r) for r in grid]
    comp = largest_component(rows)
    if len(comp) < n_players + 1:
        return rows, None, None
    cells = comp[:]
    rng.shuffle(cells)
    npc = cells[0]
    min_d = max(4, min(len(rows), len(rows[0])) // 3)
    players: list[tuple[int, int]] = []
    for c in cells[1:]:
        if len(players) == n_players:
            break
        if max(abs(c[0] - npc[0]), abs(c[1] - npc[1])) >= min_d:
            players.append(c)
    for c in cells[1:]:
        if len(players) == n_players:
            break
        if c not in players:
            players.append(c)
    return rows, npc, players


# ----------------------------------------------------------------- drawing
def draw(rows, npc, players, path=None, explored=None, vision=None, title="", path_color=NPC_COLOR, dim_path=False):
    arr = np.array([[1 if c == "#" else 0 for c in row] for row in rows])
    h, w = arr.shape
    size = max(5.0, min(9.0, max(w, h) * 0.3))
    fig, ax = plt.subplots(figsize=(size * w / max(w, h) + 0.6, size * h / max(w, h) + 0.6))
    ax.imshow(arr, cmap=ListedColormap(["#f4f1ea", "#2b2d42"]), origin="upper", interpolation="nearest")
    if explored:
        ax.scatter([p[0] for p in explored], [p[1] for p in explored], s=18, color="#ffd166", alpha=0.85, zorder=2)
    if vision:
        ax.add_patch(plt.Circle((npc[0], npc[1]), vision, fill=False, ls="--", color=NPC_COLOR, alpha=0.35, zorder=2))
    if path and len(path) > 1:
        ax.plot([p[0] for p in path], [p[1] for p in path], "-", lw=2.2, color=path_color,
                alpha=0.55 if dim_path else 0.95, zorder=3)
    for i, p in enumerate(players):
        ax.scatter(p[0], p[1], s=130, color=PLAYER_COLORS[i % len(PLAYER_COLORS)], edgecolor="white", linewidth=1.5, zorder=5)
        if len(players) > 1:
            ax.annotate(f"P{i + 1}", (p[0], p[1]), color="white", fontsize=6, ha="center", va="center", zorder=6)
    ax.scatter(npc[0], npc[1], s=150, marker="D", color=NPC_COLOR, edgecolor="white", linewidth=1.5, zorder=6)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def show(fig, container=None):
    (container or st).pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------- UI
st.set_page_config(page_title="NPC Pathfinding AI", page_icon="🎮", layout="wide")
st.title("🎮 Multiplayer Game – Pathfinding AI")
st.caption("SDG 9 · Industry, Innovation & Infrastructure  |  A* + Min-Heap + DFS wander  |  C++ backend, Streamlit frontend")

try:
    exe = build_backend()
except RuntimeError as err:
    st.error(str(err))
    st.stop()

st.session_state.setdefault("seed", 7)
with st.sidebar:
    st.header("Map")
    if st.button("🎲 New random map"):
        st.session_state["seed"] = random.randint(1, 99999)
    seed = st.number_input("Seed", min_value=1, max_value=99999, key="seed")
    map_type = st.selectbox("Map type", ["Random obstacles", "Maze (DFS generated)"])
    width = st.slider("Width", 15, 61, 31, step=2)
    height = st.slider("Height", 15, 41, 21, step=2)
    density = st.slider("Obstacle density", 0.05, 0.40, 0.20, 0.01, disabled=map_type.startswith("Maze"))
    n_players = st.slider("Players", 1, 4, 2)

    st.header("NPC AI")
    diagonal = st.checkbox("Allow diagonal moves", True)
    heuristic = st.selectbox("A* heuristic", ["euclidean", "octile", "manhattan", "zero"],
                             help="'zero' turns A* into Dijkstra. 'manhattan' is not admissible with diagonal moves.")
    vision = st.slider("Vision radius", 3, 25, 8)

rows, npc, players = build_scene(map_type, width, height, density, int(seed), n_players)
if npc is None:
    st.warning("This map has too little open space. Lower the obstacle density or change the seed.")
    st.stop()

tab_astar, tab_wander, tab_sim, tab_info = st.tabs(["🧭 A* pathfinding", "🌀 DFS wander", "🕹️ Live simulation", "📘 How it works"])

# ---------------------------------------------------------------- A* tab
with tab_astar:
    c1, c2 = st.columns([3, 2])
    with c2:
        target_idx = 0
        if n_players > 1:
            target_idx = st.selectbox("Chase which player?", list(range(n_players)), format_func=lambda i: f"Player {i + 1}")
        show_explored = st.checkbox("Show nodes A* explored", True)
    try:
        res = call_backend(exe, "astar", rows, npc=pt(npc), player=pt(players[target_idx]),
                           diag=int(diagonal), heur=heuristic, explored=int(show_explored))
        with c1:
            title = "Optimal path found" if res["found"] else "No path exists"
            show(draw(rows, npc, players, res["path"], res.get("explored"), title=title))
        with c2:
            m1, m2 = st.columns(2)
            m1.metric("Path found", "Yes" if res["found"] else "No")
            m2.metric("Path cost", f"{res['cost']:.2f}")
            m3, m4 = st.columns(2)
            m3.metric("Steps", max(0, len(res["path"]) - 1))
            m4.metric("Nodes expanded", res["expanded"])
            st.markdown("**Heuristic comparison** (same start & goal)")
            table = []
            for h in ["euclidean", "octile", "manhattan", "zero"]:
                r = call_backend(exe, "astar", rows, npc=pt(npc), player=pt(players[target_idx]),
                                 diag=int(diagonal), heur=h)
                table.append({"Heuristic": h, "Cost": round(r["cost"], 2), "Nodes expanded": r["expanded"]})
            st.table(table)
            st.caption("Fewer expanded nodes = faster search. 'zero' (Dijkstra) shows how much the heuristic helps.")
    except RuntimeError as err:
        st.error(str(err))

# ------------------------------------------------------------ wander tab
with tab_wander:
    c1, c2 = st.columns([3, 2])
    with c2:
        steps = st.slider("Wander steps", 10, 600, 150, step=10)
        wseed = st.number_input("Wander seed", 1, 99999, 1)
    try:
        res = call_backend(exe, "wander", rows, npc=pt(npc), steps=steps, seed=int(wseed))
        unique = len({tuple(p) for p in res["path"]})
        with c1:
            show(draw(rows, npc, [], res["path"], title="NPC wandering with depth-first search", path_color="#2a9d8f"))
        with c2:
            m1, m2 = st.columns(2)
            m1.metric("Steps walked", len(res["path"]) - 1)
            m2.metric("Distinct cells visited", unique)
            st.write("The NPC always walks to a random unvisited neighbour (going deeper). At a dead end it "
                     "backtracks along the DFS stack. When the whole area is explored the memory is reset.")
    except RuntimeError as err:
        st.error(str(err))

# -------------------------------------------------------- simulation tab
with tab_sim:
    c1, c2, c3 = st.columns(3)
    ticks = c1.slider("Max ticks", 20, 600, 150, step=10)
    period = c2.slider("Players move every N ticks", 1, 4, 2, help="1 = same speed as the NPC")
    sseed = c3.number_input("Simulation seed", 1, 99999, 1)
    try:
        sim = call_backend(exe, "simulate", rows, npc=pt(npc), players=":".join(pt(p) for p in players),
                           vision=vision, ticks=ticks, seed=int(sseed), diag=int(diagonal),
                           heur=heuristic, period=period)
        frames = sim["frames"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Outcome", f"Caught at tick {sim['caught_tick']}" if sim["caught"] else "Players escaped")
        m2.metric("Ticks simulated", sim["ticks"])
        m3.metric("A* searches", sim["astar_calls"])
        m4.metric("A* nodes expanded", sim["nodes_expanded"])

        tick = st.slider("Tick", 0, len(frames) - 1, 0)
        left, right = st.columns([3, 1])
        placeholder = left.empty()
        status = right.empty()

        def render(i: int) -> None:
            f = frames[i]
            label = {"CHASE": "🔴 CHASE – A* to player", "WANDER": "🟢 WANDER – DFS",
                     "CAUGHT": "💥 CAUGHT a player"}[f["mode"]]
            fig = draw(rows, f["npc"], f["players"], f["path"], vision=vision,
                       title=f"Tick {f['tick']} · {f['mode']}")
            show(fig, placeholder)
            status.markdown(f"### {label}\n\nNPC at `{tuple(f['npc'])}`")
            if f["target"] >= 0:
                status.markdown(f"Target: **Player {f['target'] + 1}**")

        if st.button("▶ Play from the start"):
            for i in range(len(frames)):
                render(i)
                time.sleep(0.08)
        else:
            render(tick)
        st.caption("Dashed circle = NPC vision radius. The NPC chases the nearest player it can see "
                   "(distance + line of sight). When nobody is visible it wanders with DFS.")
    except RuntimeError as err:
        st.error(str(err))

# -------------------------------------------------------------- info tab
with tab_info:
    st.markdown(
        """
### Architecture
```
Streamlit (app.py)  --stdin: map rows-->  C++ backend (npc_ai)
        ^                                       |
        +------------- JSON output -------------+
```
`app.py` compiles `backend/main.cpp` on first start, then calls it with `astar`, `wander` or `simulate`.

### A* search (`aStar` in `pathfinding.hpp`)
* Each cell has `g` (cost from the NPC) and `h` (estimated cost to the player).
  A* always expands the open node with the smallest **f = g + h**.
* Heuristic: **Euclidean distance** – admissible (never overestimates), so the path is optimal.
* Moves cost 1 (straight) or √2 (diagonal); diagonals may not cut corners around walls.

### Min-heap open set (`MinHeap` in `pathfinding.hpp`)
Hand-written binary heap in a `vector`: `push` and `pop` are O(log n), reading the best node is O(1).
Instead of decrease-key, improved nodes are pushed again and stale entries are skipped when popped (lazy deletion).
Complexity of A* with a binary heap: O(E log V).

### DFS wander (`DFSWanderer`)
Stack-based depth-first exploration used when no player is visible: go to a random unvisited neighbour,
backtrack at dead ends, restart when everything reachable has been visited.

### NPC behaviour (state machine)
`WANDER` --player visible--> `CHASE` --player lost--> `WANDER`  (visible = within vision radius **and** clear line of sight)

### SDG 9 link
Efficient, reusable path-planning is a building block for innovation in games, robotics and logistics infrastructure.
"""
    )

// pathfinding.hpp
// ---------------------------------------------------------------------------
// Core AI for the NPC:
//   * MinHeap          - hand-written binary min-heap (the A* "open set")
//   * aStar()          - A* search with selectable heuristic (Euclidean default)
//   * DFSWanderer      - depth-first "wander" behaviour (used when the player
//                        is not visible)
//   * hasLineOfSight() - Bresenham visibility test
//   * simulate()       - NPC (CHASE / WANDER state machine) vs. N players
// Header-only, C++17, no external dependencies.
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace npc {

constexpr double SQRT2 = 1.4142135623730951;

// ------------------------------------------------------------------ basics
struct Point {
    int x = 0;
    int y = 0;
    bool operator==(const Point& o) const { return x == o.x && y == o.y; }
    bool operator!=(const Point& o) const { return !(*this == o); }
};

// '#' is an obstacle, every other character is walkable floor.
class Grid {
public:
    int width = 0;
    int height = 0;
    std::vector<std::string> rows;

    Grid() = default;
    explicit Grid(std::vector<std::string> r) : rows(std::move(r)) {
        height = static_cast<int>(rows.size());
        for (const auto& s : rows) width = std::max(width, static_cast<int>(s.size()));
        for (auto& s : rows) s.resize(static_cast<size_t>(width), '#');  // pad short rows with walls
    }

    bool inBounds(int x, int y) const { return x >= 0 && y >= 0 && x < width && y < height; }
    bool walkable(int x, int y) const {
        return inBounds(x, y) && rows[static_cast<size_t>(y)][static_cast<size_t>(x)] != '#';
    }
    bool walkable(Point p) const { return walkable(p.x, p.y); }
    int index(Point p) const { return p.y * width + p.x; }
    Point fromIndex(int i) const { return Point{i % width, i / width}; }
};

// ----------------------------------------------------------------- MinHeap
// Binary min-heap stored in a vector. push/pop are O(log n), top is O(1).
// The element for which cmp(a, b) is true is considered "smaller" and is
// returned first.
template <typename T, typename Compare = std::less<T>>
class MinHeap {
public:
    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }

    const T& top() const {
        if (data_.empty()) throw std::out_of_range("MinHeap::top on empty heap");
        return data_.front();
    }

    void push(const T& value) {
        data_.push_back(value);
        siftUp(data_.size() - 1);
    }

    T pop() {
        if (data_.empty()) throw std::out_of_range("MinHeap::pop on empty heap");
        T result = data_.front();
        data_.front() = data_.back();
        data_.pop_back();
        if (!data_.empty()) siftDown(0);
        return result;
    }

private:
    void siftUp(size_t i) {
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (cmp_(data_[i], data_[parent])) {
                std::swap(data_[i], data_[parent]);
                i = parent;
            } else {
                break;
            }
        }
    }

    void siftDown(size_t i) {
        const size_t n = data_.size();
        for (;;) {
            size_t left = 2 * i + 1, right = left + 1, smallest = i;
            if (left < n && cmp_(data_[left], data_[smallest])) smallest = left;
            if (right < n && cmp_(data_[right], data_[smallest])) smallest = right;
            if (smallest == i) break;
            std::swap(data_[i], data_[smallest]);
            i = smallest;
        }
    }

    std::vector<T> data_;
    Compare cmp_;
};

// --------------------------------------------------------------- heuristics
enum class Heuristic { Euclidean, Manhattan, Octile, Zero };

inline Heuristic parseHeuristic(const std::string& name) {
    if (name == "euclidean") return Heuristic::Euclidean;
    if (name == "manhattan") return Heuristic::Manhattan;
    if (name == "octile") return Heuristic::Octile;
    if (name == "zero") return Heuristic::Zero;  // h = 0  ->  Dijkstra
    throw std::invalid_argument("unknown heuristic '" + name + "'");
}

inline double heuristicValue(Heuristic h, Point a, Point b) {
    const double dx = std::abs(a.x - b.x);
    const double dy = std::abs(a.y - b.y);
    switch (h) {
        case Heuristic::Euclidean: return std::sqrt(dx * dx + dy * dy);
        case Heuristic::Manhattan: return dx + dy;
        case Heuristic::Octile:    return (dx + dy) + (SQRT2 - 2.0) * std::min(dx, dy);
        case Heuristic::Zero:      return 0.0;
    }
    return 0.0;
}

// ------------------------------------------------------------------- A*
struct AStarResult {
    bool found = false;
    std::vector<Point> path;      // start ... goal (inclusive)
    double cost = 0.0;            // sum of step costs (1 or sqrt(2))
    int expanded = 0;             // nodes taken out of the open set
    std::vector<Point> explored;  // expansion order (only if requested)
};

namespace detail {
struct OpenNode {
    double f;  // g + h
    double h;  // used as tie-breaker (prefer nodes closer to the goal)
    int idx;
    bool operator<(const OpenNode& o) const {
        if (f != o.f) return f < o.f;
        return h < o.h;
    }
};
}  // namespace detail

// A* search. Open set = MinHeap keyed by f = g + h (lazy deletion: stale
// entries are skipped when popped). Diagonal moves cost sqrt(2) and are not
// allowed to cut corners around obstacles.
inline AStarResult aStar(const Grid& grid, Point start, Point goal, bool diagonal = true,
                         Heuristic heur = Heuristic::Euclidean, bool recordExplored = false) {
    AStarResult res;
    if (!grid.walkable(start) || !grid.walkable(goal)) return res;

    static const int DX[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    static const int DY[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const int dirCount = diagonal ? 8 : 4;

    const int n = grid.width * grid.height;
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> g(static_cast<size_t>(n), INF);
    std::vector<int> parent(static_cast<size_t>(n), -1);
    std::vector<char> closed(static_cast<size_t>(n), 0);

    MinHeap<detail::OpenNode> open;
    const int s = grid.index(start);
    const int t = grid.index(goal);
    const double h0 = heuristicValue(heur, start, goal);
    g[static_cast<size_t>(s)] = 0.0;
    open.push({h0, h0, s});

    while (!open.empty()) {
        const detail::OpenNode cur = open.pop();
        if (closed[static_cast<size_t>(cur.idx)]) continue;  // stale entry
        closed[static_cast<size_t>(cur.idx)] = 1;
        ++res.expanded;
        const Point p = grid.fromIndex(cur.idx);
        if (recordExplored) res.explored.push_back(p);

        if (cur.idx == t) {
            for (int c = t; c != -1; c = parent[static_cast<size_t>(c)]) res.path.push_back(grid.fromIndex(c));
            std::reverse(res.path.begin(), res.path.end());
            res.found = true;
            res.cost = g[static_cast<size_t>(t)];
            return res;
        }

        for (int d = 0; d < dirCount; ++d) {
            const int nx = p.x + DX[d], ny = p.y + DY[d];
            if (!grid.walkable(nx, ny)) continue;
            // no corner cutting on diagonals
            if (d >= 4 && !(grid.walkable(p.x + DX[d], p.y) && grid.walkable(p.x, p.y + DY[d]))) continue;
            const int ni = grid.index(Point{nx, ny});
            if (closed[static_cast<size_t>(ni)]) continue;
            const double ng = g[static_cast<size_t>(cur.idx)] + (d < 4 ? 1.0 : SQRT2);
            if (ng < g[static_cast<size_t>(ni)]) {
                g[static_cast<size_t>(ni)] = ng;
                parent[static_cast<size_t>(ni)] = cur.idx;
                const double h = heuristicValue(heur, Point{nx, ny}, goal);
                open.push({ng + h, h, ni});
            }
        }
    }
    return res;  // open set exhausted: no path
}

// --------------------------------------------------------- line of sight
// Bresenham line between two cells; every cell on the line must be walkable.
inline bool hasLineOfSight(const Grid& grid, Point a, Point b) {
    int x0 = a.x, y0 = a.y;
    const int x1 = b.x, y1 = b.y;
    const int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
    const int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        if (!grid.walkable(x0, y0)) return false;
        if (x0 == x1 && y0 == y1) return true;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

inline double euclid(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline bool canSee(const Grid& grid, Point from, Point to, double radius) {
    return euclid(from, to) <= radius && hasLineOfSight(grid, from, to);
}

// ------------------------------------------------------------ DFS wander
// Stateful, step-by-step depth-first exploration. The NPC walks the DFS tree:
// it moves to a random unvisited neighbour (going deeper); when it is stuck
// it backtracks one cell along the DFS stack. When the whole reachable region
// has been visited, the memory is cleared and the DFS restarts.
class DFSWanderer {
public:
    DFSWanderer(const Grid& grid, unsigned seed) : grid_(&grid), rng_(seed) {}

    // (Re)start the DFS from `root` with empty memory.
    void reset(Point root) {
        visited_.assign(static_cast<size_t>(grid_->width * grid_->height), 0);
        stack_.clear();
        stack_.push_back(root);
        visited_[static_cast<size_t>(grid_->index(root))] = 1;
    }

    // Returns the NPC's next position (always adjacent to, or equal to, the
    // current one).
    Point step() {
        if (stack_.empty()) throw std::logic_error("DFSWanderer::step called before reset()");
        static const int DX[4] = {1, 0, -1, 0};
        static const int DY[4] = {0, 1, 0, -1};
        for (int attempt = 0; attempt < 2; ++attempt) {
            const Point cur = stack_.back();
            std::vector<Point> candidates;
            for (int d = 0; d < 4; ++d) {
                const int nx = cur.x + DX[d], ny = cur.y + DY[d];
                if (grid_->walkable(nx, ny) && !visited_[static_cast<size_t>(grid_->index(Point{nx, ny}))])
                    candidates.push_back(Point{nx, ny});
            }
            if (!candidates.empty()) {  // go deeper
                std::shuffle(candidates.begin(), candidates.end(), rng_);
                const Point next = candidates.front();
                visited_[static_cast<size_t>(grid_->index(next))] = 1;
                stack_.push_back(next);
                return next;
            }
            if (stack_.size() > 1) {  // dead end: backtrack
                stack_.pop_back();
                return stack_.back();
            }
            reset(cur);  // everything reachable explored: start over
        }
        return stack_.back();
    }

private:
    const Grid* grid_;
    std::mt19937 rng_;
    std::vector<char> visited_;
    std::vector<Point> stack_;
};

// Convenience: wander for `steps` moves, returns positions including start.
inline std::vector<Point> wanderPath(const Grid& grid, Point start, int steps, unsigned seed) {
    DFSWanderer w(grid, seed);
    w.reset(start);
    std::vector<Point> path{start};
    for (int i = 0; i < steps; ++i) path.push_back(w.step());
    return path;
}

// ------------------------------------------------------------ simulation
struct SimConfig {
    Point npc;
    std::vector<Point> players;
    int vision = 8;
    int maxTicks = 150;
    unsigned seed = 1;
    bool diagonal = true;
    Heuristic heuristic = Heuristic::Euclidean;
    int playerPeriod = 2;  // players move once every `playerPeriod` ticks
};

struct Frame {
    int tick;
    Point npc;
    std::vector<Point> players;
    std::string mode;        // "WANDER", "CHASE" or "CAUGHT"
    std::vector<Point> path; // A* path while chasing
    int target;              // index of chased player, -1 if none
};

struct SimResult {
    std::vector<Frame> frames;
    bool caught = false;
    int caughtTick = -1;
    int astarCalls = 0;
    int nodesExpanded = 0;
};

namespace detail {
// Players do a "persistent random walk": keep going straight with 75 %
// probability, otherwise pick a random open direction.
inline Point movePlayer(const Grid& g, Point p, int& dir, std::mt19937& rng) {
    static const int DX[4] = {1, 0, -1, 0};
    static const int DY[4] = {0, 1, 0, -1};
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (u(rng) < 0.75 && g.walkable(p.x + DX[dir], p.y + DY[dir])) return Point{p.x + DX[dir], p.y + DY[dir]};
    std::vector<int> open;
    for (int d = 0; d < 4; ++d)
        if (g.walkable(p.x + DX[d], p.y + DY[d])) open.push_back(d);
    if (open.empty()) return p;
    dir = open[rng() % open.size()];
    return Point{p.x + DX[dir], p.y + DY[dir]};
}
}  // namespace detail

inline SimResult simulate(const Grid& grid, const SimConfig& cfg) {
    SimResult out;
    std::mt19937 prng(cfg.seed * 7919u + 13u);
    DFSWanderer wanderer(grid, cfg.seed);

    Point npc = cfg.npc;
    std::vector<Point> players = cfg.players;
    std::vector<int> dirs(players.size());
    for (auto& d : dirs) d = static_cast<int>(prng() % 4);

    wanderer.reset(npc);
    bool wasChasing = false;
    out.frames.push_back(Frame{0, npc, players, "WANDER", {}, -1});

    auto caughtIndex = [&]() {
        for (size_t i = 0; i < players.size(); ++i)
            if (players[i] == npc) return static_cast<int>(i);
        return -1;
    };

    for (int tick = 1; tick <= cfg.maxTicks; ++tick) {
        // 1) players move
        if (tick % std::max(1, cfg.playerPeriod) == 0)
            for (size_t i = 0; i < players.size(); ++i)
                players[i] = detail::movePlayer(grid, players[i], dirs[i], prng);

        int hit = caughtIndex();
        if (hit >= 0) {
            out.caught = true;
            out.caughtTick = tick;
            out.frames.push_back(Frame{tick, npc, players, "CAUGHT", {}, hit});
            break;
        }

        // 2) NPC picks the nearest visible player
        int target = -1;
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < players.size(); ++i) {
            if (canSee(grid, npc, players[i], cfg.vision)) {
                const double d = euclid(npc, players[i]);
                if (d < best) { best = d; target = static_cast<int>(i); }
            }
        }

        // 3) NPC acts: CHASE (A*) or WANDER (DFS)
        if (target >= 0) {
            AStarResult r = aStar(grid, npc, players[static_cast<size_t>(target)], cfg.diagonal, cfg.heuristic);
            ++out.astarCalls;
            out.nodesExpanded += r.expanded;
            if (r.found && r.path.size() > 1) npc = r.path[1];
            wasChasing = true;
            out.frames.push_back(Frame{tick, npc, players, "CHASE", r.path, target});
        } else {
            if (wasChasing) {  // lost the player: restart DFS from here
                wanderer.reset(npc);
                wasChasing = false;
            }
            npc = wanderer.step();
            out.frames.push_back(Frame{tick, npc, players, "WANDER", {}, -1});
        }

        hit = caughtIndex();
        if (hit >= 0) {
            out.caught = true;
            out.caughtTick = tick;
            out.frames.back().mode = "CAUGHT";
            out.frames.back().target = hit;
            break;
        }
    }
    return out;
}

}  // namespace npc

// main.cpp - command-line backend for the NPC pathfinding project.
//
// The Streamlit frontend runs this program and exchanges JSON with it.
// The grid is read from stdin (or --map FILE): one row per line,
// '#' = obstacle, anything else = floor.
//
//   npc_ai astar    --npc x,y --player x,y [--diag 1] [--heur euclidean] [--explored 0]
//   npc_ai wander   --npc x,y [--steps 60] [--seed 1]
//   npc_ai simulate --npc x,y --players x,y:x,y [--vision 8] [--ticks 150]
//                   [--seed 1] [--diag 1] [--heur euclidean] [--period 2]
//   npc_ai test     (runs the built-in self tests)

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "pathfinding.hpp"

using namespace npc;

// ------------------------------------------------------------ JSON helpers
static std::string num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.4f", v);
    return buf;
}

static std::string pointJson(Point p) { return "[" + std::to_string(p.x) + "," + std::to_string(p.y) + "]"; }

static std::string pathJson(const std::vector<Point>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += pointJson(v[i]);
    }
    return s + "]";
}

static std::string jsonEscape(const std::string& in) {
    std::string out;
    for (char c : in) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

// ------------------------------------------------------------ arg parsing
using Args = std::map<std::string, std::string>;

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 2; i < argc; i += 2) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) throw std::runtime_error("expected an option starting with -- but got '" + key + "'");
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        a[key.substr(2)] = argv[i + 1];
    }
    return a;
}

static std::string getStr(const Args& a, const std::string& k, const std::string& def) {
    auto it = a.find(k);
    return it == a.end() ? def : it->second;
}

static int getInt(const Args& a, const std::string& k, int def) {
    auto it = a.find(k);
    if (it == a.end()) return def;
    try { return std::stoi(it->second); }
    catch (...) { throw std::runtime_error("option --" + k + " needs an integer"); }
}

static Point parsePoint(const std::string& s) {
    int x, y;
    char extra;
    if (std::sscanf(s.c_str(), "%d,%d%c", &x, &y, &extra) != 2) throw std::runtime_error("bad point '" + s + "', expected x,y");
    return Point{x, y};
}

static std::vector<Point> parsePoints(const std::string& s) {
    std::vector<Point> pts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ':'))
        if (!item.empty()) pts.push_back(parsePoint(item));
    return pts;
}

static Point requirePoint(const Args& a, const std::string& k) {
    auto it = a.find(k);
    if (it == a.end()) throw std::runtime_error("missing required option --" + k);
    return parsePoint(it->second);
}

static Grid readGrid(const Args& a) {
    std::vector<std::string> rows;
    std::string line;
    auto consume = [&](std::istream& in) {
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (!line.empty()) rows.push_back(line);
        }
    };
    auto it = a.find("map");
    if (it != a.end()) {
        std::ifstream f(it->second);
        if (!f) throw std::runtime_error("cannot open map file " + it->second);
        consume(f);
    } else {
        consume(std::cin);
    }
    if (rows.empty()) throw std::runtime_error("the map is empty");
    return Grid(rows);
}

static void requireWalkable(const Grid& g, Point p, const std::string& what) {
    if (!g.inBounds(p.x, p.y)) throw std::runtime_error(what + " is outside the map");
    if (!g.walkable(p)) throw std::runtime_error(what + " is standing on an obstacle");
}

// ---------------------------------------------------------------- commands
static std::string cmdAStar(const Args& a) {
    Grid grid = readGrid(a);
    Point npcPos = requirePoint(a, "npc"), player = requirePoint(a, "player");
    requireWalkable(grid, npcPos, "NPC");
    requireWalkable(grid, player, "player");
    const bool diag = getInt(a, "diag", 1) != 0;
    const bool explored = getInt(a, "explored", 0) != 0;
    Heuristic h = parseHeuristic(getStr(a, "heur", "euclidean"));

    AStarResult r = aStar(grid, npcPos, player, diag, h, explored);
    std::string json = "{\"found\":" + std::string(r.found ? "true" : "false") + ",\"cost\":" + num(r.cost) +
                       ",\"expanded\":" + std::to_string(r.expanded) + ",\"path\":" + pathJson(r.path);
    if (explored) json += ",\"explored\":" + pathJson(r.explored);
    return json + "}";
}

static std::string cmdWander(const Args& a) {
    Grid grid = readGrid(a);
    Point npcPos = requirePoint(a, "npc");
    requireWalkable(grid, npcPos, "NPC");
    int steps = std::max(1, std::min(5000, getInt(a, "steps", 60)));
    unsigned seed = static_cast<unsigned>(getInt(a, "seed", 1));
    return "{\"path\":" + pathJson(wanderPath(grid, npcPos, steps, seed)) + "}";
}

static std::string cmdSimulate(const Args& a) {
    Grid grid = readGrid(a);
    SimConfig cfg;
    cfg.npc = requirePoint(a, "npc");
    requireWalkable(grid, cfg.npc, "NPC");
    auto pit = a.find("players");
    if (pit == a.end()) throw std::runtime_error("missing required option --players");
    cfg.players = parsePoints(pit->second);
    if (cfg.players.empty() || cfg.players.size() > 8) throw std::runtime_error("--players needs 1 to 8 positions");
    for (size_t i = 0; i < cfg.players.size(); ++i) requireWalkable(grid, cfg.players[i], "player " + std::to_string(i + 1));
    cfg.vision = std::max(1, getInt(a, "vision", 8));
    cfg.maxTicks = std::max(1, std::min(5000, getInt(a, "ticks", 150)));
    cfg.seed = static_cast<unsigned>(getInt(a, "seed", 1));
    cfg.diagonal = getInt(a, "diag", 1) != 0;
    cfg.heuristic = parseHeuristic(getStr(a, "heur", "euclidean"));
    cfg.playerPeriod = std::max(1, getInt(a, "period", 2));

    SimResult r = simulate(grid, cfg);
    std::string json = "{\"caught\":" + std::string(r.caught ? "true" : "false") +
                       ",\"caught_tick\":" + std::to_string(r.caughtTick) +
                       ",\"ticks\":" + std::to_string(r.frames.size() - 1) +
                       ",\"astar_calls\":" + std::to_string(r.astarCalls) +
                       ",\"nodes_expanded\":" + std::to_string(r.nodesExpanded) + ",\"frames\":[";
    for (size_t i = 0; i < r.frames.size(); ++i) {
        const Frame& f = r.frames[i];
        if (i) json += ",";
        json += "{\"tick\":" + std::to_string(f.tick) + ",\"npc\":" + pointJson(f.npc) +
                ",\"players\":" + pathJson(f.players) + ",\"mode\":\"" + f.mode + "\",\"target\":" +
                std::to_string(f.target) + ",\"path\":" + pathJson(f.path) + "}";
    }
    return json + "]}";
}

// -------------------------------------------------------------- self tests
static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL (line " << __LINE__ << "): " #cond << std::endl;  \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

static int cmdTest() {
    // MinHeap returns elements in sorted order
    {
        MinHeap<int> h;
        for (int v : {5, 1, 4, 2, 3, 9, 0}) h.push(v);
        std::vector<int> out;
        while (!h.empty()) out.push_back(h.pop());
        CHECK((out == std::vector<int>{0, 1, 2, 3, 4, 5, 9}));
    }
    // open 5x5 grid
    {
        Grid g({".....", ".....", ".....", ".....", "....."});
        AStarResult d = aStar(g, {0, 0}, {4, 4}, true);
        CHECK(d.found && d.path.size() == 5 && std::abs(d.cost - 4 * SQRT2) < 1e-9);
        AStarResult f = aStar(g, {0, 0}, {4, 4}, false);
        CHECK(f.found && std::abs(f.cost - 8.0) < 1e-9);
        AStarResult same = aStar(g, {2, 2}, {2, 2});
        CHECK(same.found && same.path.size() == 1 && same.cost == 0.0);
    }
    // fully walled off -> no path
    {
        Grid g({"..#..", "..#..", "..#.."});
        CHECK(!aStar(g, {0, 0}, {4, 0}).found);
        CHECK(!aStar(g, {2, 0}, {4, 0}).found);  // start inside a wall
    }
    // detour around a wall
    {
        Grid g({"..#..", "..#..", "....."});
        AStarResult r = aStar(g, {0, 0}, {4, 0}, false);
        CHECK(r.found && std::abs(r.cost - 8.0) < 1e-9);
    }
    // no corner cutting between two diagonal obstacles
    {
        Grid g({".#", "#."});
        CHECK(!aStar(g, {0, 0}, {1, 1}, true).found);
    }
    // line of sight
    {
        Grid g({".....", "..#..", "....."});
        CHECK(!hasLineOfSight(g, {0, 1}, {4, 1}));
        CHECK(hasLineOfSight(g, {0, 0}, {4, 0}));
    }
    // property test: admissible heuristics must match Dijkstra (h = 0)
    {
        std::mt19937 rng(12345);
        int compared = 0;
        for (int t = 0; t < 200; ++t) {
            const int w = 12, h = 10;
            std::vector<std::string> rows(h, std::string(w, '.'));
            for (auto& r : rows)
                for (auto& c : r)
                    if (rng() % 100 < 28) c = '#';
            Grid g(rows);
            Point a{static_cast<int>(rng() % w), static_cast<int>(rng() % h)};
            Point b{static_cast<int>(rng() % w), static_cast<int>(rng() % h)};
            for (bool diag : {true, false}) {
                AStarResult ref = aStar(g, a, b, diag, Heuristic::Zero);
                for (Heuristic hh : {Heuristic::Euclidean, Heuristic::Octile}) {
                    AStarResult r = aStar(g, a, b, diag, hh);
                    CHECK(r.found == ref.found);
                    if (r.found && ref.found) {
                        CHECK(std::abs(r.cost - ref.cost) < 1e-9);
                        ++compared;
                        for (size_t i = 1; i < r.path.size(); ++i) {
                            CHECK(g.walkable(r.path[i]));
                            CHECK(std::abs(r.path[i].x - r.path[i - 1].x) <= 1 && std::abs(r.path[i].y - r.path[i - 1].y) <= 1);
                        }
                    }
                }
            }
        }
        CHECK(compared > 100);
    }
    // DFS wander: legal, adjacent steps, and it actually explores
    {
        Grid g({"........", "..####..", "........", ".#....#.", "........"});
        std::vector<Point> p = wanderPath(g, {0, 0}, 300, 7);
        CHECK(p.size() == 301);
        bool allOk = true;
        std::vector<Point> uniq;
        for (size_t i = 0; i < p.size(); ++i) {
            if (!g.walkable(p[i])) allOk = false;
            if (i && std::abs(p[i].x - p[i - 1].x) + std::abs(p[i].y - p[i - 1].y) > 1) allOk = false;
            if (std::find(uniq.begin(), uniq.end(), p[i]) == uniq.end()) uniq.push_back(p[i]);
        }
        CHECK(allOk);
        CHECK(uniq.size() > 20);
        // a single isolated cell must not crash
        Grid lone({".#", "##"});
        CHECK(wanderPath(lone, {0, 0}, 5, 1).size() == 6);
    }
    // simulation: terminates, NPC steps are legal, a visible player gets caught
    {
        Grid g({"..........", "..........", "..........", "..........", ".........."});
        SimConfig cfg;
        cfg.npc = {0, 0};
        cfg.players = {{6, 3}};
        cfg.vision = 20;
        cfg.maxTicks = 100;
        cfg.playerPeriod = 3;
        SimResult r = simulate(g, cfg);
        CHECK(r.caught);
        CHECK(r.astarCalls > 0);
        bool ok = true;
        for (size_t i = 1; i < r.frames.size(); ++i) {
            const Point a = r.frames[i - 1].npc, b = r.frames[i].npc;
            if (std::abs(a.x - b.x) > 1 || std::abs(a.y - b.y) > 1) ok = false;
        }
        CHECK(ok);
    }
    if (failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << failures << " TEST(S) FAILED" << std::endl;
    return 1;
}

// -------------------------------------------------------------------- main
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: npc_ai <astar|wander|simulate|test> [options]");
        const std::string cmd = argv[1];
        if (cmd == "test") return cmdTest();
        Args args = parseArgs(argc, argv);
        std::string out;
        if (cmd == "astar") out = cmdAStar(args);
        else if (cmd == "wander") out = cmdWander(args);
        else if (cmd == "simulate") out = cmdSimulate(args);
        else throw std::runtime_error("unknown command '" + cmd + "'");
        std::cout << out << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "{\"error\":\"" << jsonEscape(e.what()) << "\"}" << std::endl;
        return 1;
    }
}

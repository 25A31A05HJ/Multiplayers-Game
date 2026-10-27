CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

npc_ai: main.cpp pathfinding.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o npc_ai

test: npc_ai
	./npc_ai test

clean:
	rm -f npc_ai npc_ai.exe

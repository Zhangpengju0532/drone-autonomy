import heapq
import itertools
import math

import numpy as np

import world

MOVES = [m for m in itertools.product((-1, 0, 1), repeat=3) if m != (0, 0, 0)]
COSTS = [math.sqrt(sum(v * v for v in m)) for m in MOVES]


def search(grid, start, goal, use_heuristic=True):
    def guess(cell):
        if not use_heuristic:
            return 0.0
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(cell, goal)))

    best = {start: 0.0}
    parent = {}
    queue = [(guess(start), 0.0, start)]
    closed = set()
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cell in closed:
            continue
        closed.add(cell)
        if cell == goal:
            path = [cell]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return path[::-1], len(closed)
        for move, step in zip(MOVES, COSTS):
            nxt = (cell[0] + move[0], cell[1] + move[1], cell[2] + move[2])
            if min(nxt) < 0 or nxt[0] >= grid.shape[0] or nxt[1] >= grid.shape[1] or nxt[2] >= grid.shape[2]:
                continue
            if grid[nxt] or nxt in closed:
                continue
            new_cost = cost + step
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                parent[nxt] = cell
                heapq.heappush(queue, (new_cost + guess(nxt), new_cost, nxt))
    return None, len(closed)


def clear_line(grid, a, b):
    length = float(np.linalg.norm(b - a))
    for s in np.linspace(0.0, 1.0, max(2, int(length / (world.CELL / 4)) + 1)):
        cell = world.cell_of(a + (b - a) * s)
        if grid[cell]:
            return False
    return True


def shortcut(grid, points):
    kept = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not clear_line(grid, points[i], points[j]):
            j -= 1
        kept.append(points[j])
        i = j
    return kept


def length(points):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:])))


def plan(start=None, goal=None):
    grid = world.occupancy()
    start = world.START if start is None else start
    goal = world.GOAL if goal is None else goal
    cells, expanded = search(grid, world.cell_of(start), world.cell_of(goal))
    if cells is None:
        return None
    points = [np.array(start, dtype=float)] + [world.centre_of(c) for c in cells[1:-1]] + [np.array(goal, dtype=float)]
    return shortcut(grid, points), points, expanded
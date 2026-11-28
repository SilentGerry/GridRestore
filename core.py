"""
core.py - THE ALGORITHM ENGINE  (CPSC 335 Final Project)

This is the file you complete. Every project option (GridRestore, ShipSmart,
BloodLink) uses these SAME functions, so the app works as soon as they do.

    Part 1  build_index                      hash table + duplicate check
    Part 2  merge_sort, lower_bound          divide & conquer sort, binary search
    Part 3  MinHeap                          priority queue
    Part 4  greedy_select                    greedy algorithm
    Part 5  dp_select                        dynamic programming (0/1 knapsack)
    Part 6  bfs_reachable, dijkstra, mst_kruskal   graph algorithms
    Part 7  kmp_search                       string matching
    Part 8  tour_brute_force, tour_heuristic an NP-hard problem (TSP)

RULES (checked by check_project.py):
  * Write the algorithms yourself. Inside this file do NOT use:
    sorted(), .sort(), heapq, bisect, str.find(), str.index(), str.count(), re.
  * dict, set, list, collections.deque and itertools.permutations ARE allowed.
  * Do not change any function name or its inputs/outputs. The app and the
    tests depend on them.

Delete each `raise NotImplementedError(...)` line when you write that function.
Run  python check_project.py  to see which parts pass.
"""

from collections import deque
from itertools import permutations
import time


# =====================================================================
# PART 1 - Hash table lookup                              (Weeks 2, 11)
# =====================================================================
def build_index(records, key):
    """Index records by record[key] and find duplicate keys.

    records : list of dicts, e.g. [{"id": "A", ...}, {"id": "B", ...}]
    key     : the field to index by, e.g. "id"

    Return (index, duplicates)
      index      : dict  key value -> record   (keep the FIRST record seen)
      duplicates : list of key values seen more than once, in the order found

    Example: records with ids A, B, A  ->  ({"A": ..., "B": ...}, ["A"])
    Target: expected O(n) time.
    """

    indexed = {}
    duplicates = []

    for record in records:
        #print(f"RECORD: {record[0]}")
        if record[key] not in indexed:
            indexed[record[key]] = record
        else:
            if record[key] not in duplicates:
                duplicates.append(record[key])

    return indexed, duplicates

    #raise NotImplementedError("Part 1: build_index")


# =====================================================================
# PART 2 - Merge sort and binary search                        (Week 4)
# =====================================================================
def merge_sort(items, key=lambda x: x):
    """Return a NEW list with the items sorted by key(item), smallest first.

    Must be STABLE: items with equal keys keep their original order.
    Steps: 1) a list of 0 or 1 items is already sorted
           2) split in half and merge_sort each half (recursion)
           3) merge the two sorted halves: repeatedly take the smaller front item
              (when the keys are equal, take from the LEFT half -> stable)
    Target: O(n log n).
    """

    size = len(items)
    if size <= 1:
        return items

    middle = size // 2
    left_items = merge_sort(items[:middle], key)
    right_items = merge_sort(items[middle:], key)
    merge_sorted = []
    left_count = 0
    right_count = 0

    while len(left_items) > left_count and len(right_items) > right_count:
        if key(left_items[left_count]) <= key(right_items[right_count]):
            merge_sorted.append(left_items[left_count])
            left_count += 1
        else:
            merge_sorted.append(right_items[right_count])
            right_count += 1

    merge_sorted.extend(left_items[left_count:])
    merge_sorted.extend(right_items[right_count:])


    return merge_sorted
    #raise NotImplementedError("Part 2: merge_sort")


def lower_bound(sorted_items, target, key=lambda x: x):
    """Binary search on a list already sorted by key.

    Return the FIRST index i where key(sorted_items[i]) >= target.
    If every item is smaller than target, return len(sorted_items).

    Example: [1, 3, 3, 5, 9] with target 3 -> 1,  target 4 -> 3,  target 10 -> 5
    Target: O(log n).  Keep two pointers lo, hi and halve the range each step.
    """

    if len(sorted_items) == 0:
        return 0

    start = 0
    end = len(sorted_items)
    middle = 0

    while start < end:
        middle = (start + end) // 2
        if key(sorted_items[middle]) >= target:
            end = middle
        else:
            start = middle + 1

    return start
    #raise NotImplementedError("Part 2: lower_bound")


# =====================================================================
# PART 3 - Priority queue (binary min-heap)                    (Week 11)
# =====================================================================
class MinHeap:
    """A binary min-heap of (priority, item) pairs stored in a Python list.

    The SMALLEST priority comes out first. Priorities may be numbers or tuples.
    When two priorities are equal, the item pushed FIRST comes out first.
    Hint: store entries as (priority, counter, item) and compare (priority, counter).

    For index i: parent = (i - 1) // 2, children = 2*i + 1 and 2*i + 2.
    """

    def __init__(self):
        self._a = []          # the heap array
        self._count = 0       # insertion counter for tie-breaking

    def __len__(self):
        return len(self._a)

    def push(self, priority, item):
        """Add an item. Append at the end, then 'bubble up'. O(log n)."""

        self._a.append((priority, self._count, item))
        self._count += 1

        size = len(self._a)
        index = size - 1
        parent_index = (index - 1) // 2

        #print(f"\n---PUSH----\nPIORITY: {priority}\nITEM: {item}")

        while index > 0 and self._a[index][0] < self._a[parent_index][0]:
            temp_index = self._a[index]
            self._a[index] = self._a[parent_index]
            self._a[parent_index] = temp_index
            index = parent_index
            parent_index = (index - 1) // 2

        #raise NotImplementedError("Part 3: MinHeap.push")

    def peek(self):
        """Return (priority, item) with the smallest priority without removing it.
        Raise IndexError if the heap is empty. O(1)."""
        if len(self._a) == 0:
            raise IndexError("MinHeap is empty.")

        return (self._a[0][0], self._a[0][2])
        #raise NotImplementedError("Part 3: MinHeap.peek")

    def pop(self):
        """Remove and return (priority, item) with the smallest priority.
        Move the last entry to the top, then 'bubble down'.
        Raise IndexError if the heap is empty. O(log n)."""

        if len(self._a) == 0:
            raise IndexError("MinHeap is empty.")

        min = (self._a[0][0], self._a[0][2])
        self._a[0] = self._a[-1]
        self._a = self._a[:-1]

        size = len(self._a)
        current_index = 0
        while True:
            parent_index = current_index
            left_child_index = current_index * 2 + 1
            right_child_index = current_index * 2 + 2

            if left_child_index < size and self._a[left_child_index] < self._a[parent_index]:
                parent_index = left_child_index
            if right_child_index < size and self._a[right_child_index] < self._a[parent_index]:
                parent_index = right_child_index
            if parent_index != current_index:
                temp_current = self._a[current_index]
                self._a[current_index] = self._a[parent_index]
                self._a[parent_index] = temp_current
                current_index = parent_index
            else:
                break

        return min
        #raise NotImplementedError("Part 3: MinHeap.pop")


# =====================================================================
# PART 4 - Greedy                                               (Week 5)
# =====================================================================
def greedy_select(items, capacity):
    """Greedy choice: best value per unit of weight first.

    items    : list of dicts {"id": ..., "weight": positive int, "value": number}
    capacity : int, the most total weight you may take

    1) order the items by value / weight, highest first (use YOUR merge_sort;
       equal ratios keep their original order)
    2) walk that order and take every item that still fits

    Return (chosen_ids, total_value, total_weight), chosen_ids in the order taken.
    Target: O(n log n).
    """
    raise NotImplementedError("Part 4: greedy_select")


# =====================================================================
# PART 5 - Dynamic programming: 0/1 knapsack                (Weeks 6-7)
# =====================================================================
def dp_select(items, capacity):
    """Same input and output as greedy_select, but always the BEST total value.

    Table: best[i][c] = best value using only the first i items with capacity c
      best[0][c] = 0
      best[i][c] = best[i-1][c]                                  (skip item i)
                   or best[i-1][c - w] + v   if w <= c           (take item i)
    Then walk back from best[n][capacity] to find which items were taken.

    Return (chosen_ids, total_value, total_weight). chosen_ids in the same order
    as the items list. Target: O(n * capacity) time.
    """
    raise NotImplementedError("Part 5: dp_select")


# =====================================================================
# PART 6 - Graphs                                           (Weeks 9-10)
# graph format: dict  node -> list of (neighbor, weight), undirected
# =====================================================================
def bfs_reachable(graph, start, blocked_edge=None):
    """Breadth-first search from start.

    blocked_edge : None, or a pair (u, v) that may NOT be used in either
                   direction (for example a broken line or a closed road).

    Return a dict  node -> number of hops from start  for every reachable node
    (start itself has 0). Use collections.deque as the queue. Target: O(V + E).
    """
    raise NotImplementedError("Part 6: bfs_reachable")


def dijkstra(graph, start):
    """Shortest paths from start (all weights >= 0). Use YOUR MinHeap.

    Return (dist, prev)
      dist[node] : length of the shortest path from start
      prev[node] : the node just before it on that path (prev[start] = None)
    Only reachable nodes appear in dist and prev. Target: O((V + E) log V).
    """
    raise NotImplementedError("Part 6: dijkstra")


def path_to(prev, target):
    """GIVEN (do not change). Rebuild the path start -> target from prev."""
    if target not in prev:
        return []
    path = []
    while target is not None:
        path.append(target)
        target = prev[target]
    path.reverse()
    return path


def mst_kruskal(nodes, edges):
    """Minimum spanning tree with Kruskal's algorithm + union-find.

    nodes : list of node ids
    edges : list of (u, v, cost)

    Sort the edges by cost (YOUR merge_sort), then add each edge whose ends are
    in different components. Return (tree_edges, total_cost), tree_edges as
    (u, v, cost) in the order added. Target: O(E log E).
    """
    raise NotImplementedError("Part 6: mst_kruskal")


# =====================================================================
# PART 7 - String matching                                     (Week 12)
# =====================================================================
def kmp_search(text, pattern):
    """Knuth-Morris-Pratt. Return a list of EVERY start index where pattern
    occurs in text, overlapping matches included. Empty pattern -> [].

    Example: kmp_search("aaaa", "aa") -> [0, 1, 2]
    1) build the failure table: fail[i] = length of the longest proper prefix of
       pattern[:i+1] that is also a suffix of it
    2) scan the text once, never moving backwards in the text
    Target: O(n + m).  Do not use str.find, str.index, `in` on strings, or re.
    """

    pattern_size = len(pattern)

    if pattern_size == 0:
        return []

    fail = [0] * pattern_size
    length = 0
    index = 1

    while index < pattern_size:
        if pattern[index] == pattern[length]:
            length += 1
            fail[index] = length
            index += 1
        elif length > 0:
            length = fail[length - 1]
        else:
            fail[index] = 0
            index += 1

    pattern_index = 0
    text_index = 0
    matches = []
    text_size = len(text)

    while text_index < text_size:
        if text[text_index] == pattern[pattern_index]:
            text_index += 1
            pattern_index += 1
            if pattern_index == pattern_size:
                matches.append(text_index - pattern_index)
                pattern_index = fail[pattern_index - 1]
        elif pattern_index > 0:
            pattern_index = fail[pattern_index - 1]
        else:
            text_index += 1

    return matches
    #raise NotImplementedError("Part 7: kmp_search")


# =====================================================================
# PART 8 - An NP-hard problem: the shortest tour (TSP)          (Week 14)
# dist format: dist[a][b] = travel cost from place a to place b
# =====================================================================
def tour_cost(order, dist):
    """GIVEN (do not change). Cost of order[0] -> ... -> order[-1] -> order[0]."""
    return sum(dist[order[i]][order[(i + 1) % len(order)]] for i in range(len(order)))


def tour_brute_force(start, stops, dist):
    """EXACT answer: try every order of the stops (itertools.permutations).

    Each tour begins at start, visits every stop once, and returns to start.
    Return (order, cost) where order is a list beginning with start
    (do not repeat start at the end). If stops is empty return ([start], 0).
    Target: O(n!) - only for about 9 stops or fewer.
    """

    if len(stops) == 0:
        return ([start], 0)

    routes = list(permutations(stops))
    shortest = ([start] + list(routes[0]), tour_cost([start] + list(routes[0]), dist))

    for route in routes[1:]:
        curr_cost = tour_cost([start] + list(route), dist)
        if curr_cost < shortest[1]:
            shortest = ([start] + list(route), curr_cost)

    return shortest

    #raise NotImplementedError("Part 8: tour_brute_force")


def tour_heuristic(start, stops, dist):
    """FAST answer: nearest neighbor, then 2-opt improvement.

    1) nearest neighbor: from start, always go to the closest unvisited stop
    2) 2-opt: try reversing every segment order[i..j] (i >= 1, so start stays
       first); keep a reversal if it makes the tour shorter; repeat until no
       reversal helps
    Return (order, cost) in the same format as tour_brute_force.
    """

    route = [start]
    curr_stop = start
    size = len(stops)

    for i in range(size):
        sorted_neighbors = merge_sort([x for x in dist[curr_stop].items() if x[0] not in route and x[0] in stops], key=lambda item: item[1])
        route.append(sorted_neighbors[0][0])

    shortest_route = route
    shortest_cost = tour_cost(route, dist)
    better = True

    while better:
        better = False
        for start in range(1,size - 1):
            for end in range(start + 1, size - 1):
                candidate_route = shortest_route[:start] + shortest_route[start:end + 1][::-1] + shortest_route[end + 1:]
                candidate_cost = tour_cost(candidate_route, dist)
                if candidate_cost < shortest_cost:
                    shortest_route = candidate_route
                    shortest_cost = candidate_cost
                    better = True
                    break
            if better:
                break

    return (shortest_route, shortest_cost)
    #raise NotImplementedError("Part 8: tour_heuristic")


# =====================================================================
# GIVEN - timing helper for the speed report               (Weeks 3, 13)
# =====================================================================
def time_it(func, *args, repeat=3):
    """GIVEN (do not change). Best wall-clock time in seconds over `repeat` runs."""
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        t = time.perf_counter() - t0
        best = t if best is None or t < best else best
    return best

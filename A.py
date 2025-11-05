import heapq


graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 1), ('E', 4)],
    'C': [('F', 2)],
    'D': [],
    'E': [],
    'F': []
}


heuristic = {
    'A': 5,
    'B': 3,
    'C': 2,
    'D': 6,
    'E': 4,
    'F': 0
}

def a_star(start, goal):
    pq = [(heuristic[start], 0, start, [start])]  
    visited = set()

    while pq:
        f, g, node, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            print("Path found:", path, "with cost:", g)
            return

        for neighbor, cost in graph[node]:
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))

    print("No path found")

a_star('A', 'F')

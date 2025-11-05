from queue import PriorityQueue

class Graph:
    def __init__(self):
        self.graph = {} 

    def addEdge(self, u, v, cost):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append((v, cost))

    def best_first_search(self, start, goal, heuristic):
        visited = set()
        pq = PriorityQueue()
        pq.put((heuristic[start], start))
        
        print("Best First Search Path:")
        while not pq.empty():
            h, current = pq.get()
            print(current, end=' ')
            if current == goal:
                break
            visited.add(current)
            for neighbor, cost in self.graph.get(current, []):
                if neighbor not in visited:
                    pq.put((heuristic[neighbor], neighbor))


g = Graph()
g.addEdge('A', 'B', 1)
g.addEdge('A', 'C', 1)
g.addEdge('B', 'D', 1)
g.addEdge('C', 'E', 1)
g.addEdge('D', 'F', 1)
g.addEdge('E', 'F', 1)

heuristic = {
    'A': 5,
    'B': 4,
    'C': 3,
    'D': 2,
    'E': 1,
    'F': 0
}

g.best_first_search('A', 'F', heuristic)

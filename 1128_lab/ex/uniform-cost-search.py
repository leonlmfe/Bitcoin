import heapq

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def successor(self):
        result = []
        result.append(State(self.x + 1, self.y))
        result.append(State(self.x , self.y + 1))
        result.append(State(self.x, self.y - 1))
        result.append(State(self.x - 1, self.y))
        return result;

def uniform_cost_search(start, goal):
    visit = {}
    parent_node = {}
    path_cost = {}
    path_cost[start] = 0
    
    frontier = []
    heapq.heappush(frontier, (0, start))

    while True:
        if not frontier:
            return "Error"
        
        x_cost, x = heapq.heappop(frontier)
        visit[x] = True

        if x == goal:
            result = []
            t = x
            while t != start:
                result.append(t)
                t = parent_node[t]
            return result
        
        for u in x.successor():
            if not visit[u]:
                parent_node[u] = x
                u_cost = x_cost + cost_compute(x, u)
                path_cost[u] = u_cost
                heapq.heappush(frontier, (u_cost, u))
            else:
                u_cost = x_cost + cost_compute(x, u)
                if u_cost < path_cost[u]:
                    parent_node[u] = x
                    path_cost[u] = u_cost
                    heapq.heappush(frontier, (u_cost, u))
                    
                









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

def h(current, target):
    pass

def astar_search(start, goal):
    visit = {}
    parent_node = {}
    path_cost = {}
    path_cost[start] = 0

    initial_h_value = h(start, goal)
    frontier = []
    heapq.heappush(frontier, (0 + initial_h_value, initial_h_value, 0, start))

    while True:
        if not frontier:
            return "Error"
        
        x_f, x_h, x_g, x = heapq.heappop(frontier)
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
                u_g = x_g + cost_compute(x, u)
                path_cost[u] = u_g
                u_h = h(u, goal)
                heapq.heappush(frontier, (u_g + u_h, u_h, u_g, u))

            


                
                    

                









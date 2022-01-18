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

def bibf_search(start, goal):
    visit_start = {}
    visit_goal = {}

    parent_node_from_start = {}
    parent_node_from_goal = {}
    path_cost_from_start = {}
    path_cost_from_start[start] = 0
    path_cost_from_goal = {}
    path_cost_from_goal[goal] = 0
    
    frontier_from_start = []
    heapq.heappush(frontier_from_start, (0, start))

    frontier_from_goal = []
    heapq.heappush(frontiter_from_goal, (0, goal))

    while True:
        if not frontier_from_goal and not frontier_from_goal:
            return "Error"
        
        x_cost, x = heapq.heappop(frontier_from_start)
        visit_start[x] = True

        if visit_goal[x]:
            result_to_start = []
            result_to_goal = []

            t = x
            while t != start:
                result_to_start.append(t)
                t = parent_node_from_start[t]

            t = x
            while t != goal:
                result_to_goal.append(t)
                t = parent_node_from_goal[t]
            return result_to_start.reverse() + result_to_goal
        
        for u in x.successor():
            if not visit_start[u]:
                parent_node_from_start[u] = x
                u_cost = x_cost + cost_compute(x, u)
                path_cost_from_start[u] = u_cost
                heapq.heappush(frontier_from_start, (u_cost, u))
            else:
                u_cost = x_cost + cost_compute(x, u)
                if u_cost < path_cost_from_start[u]:
                    parent_node_from_start[u] = x
                    path_cost_from_start[u] = u_cost
                    heapq.heappush(frontier_from_start, (u_cost, u))

        # step from goal 
        x_cost, x = heapq.heappop(frontier_from_start)
        visit_goal[x] = True

        if visit_start[x]:
            result_to_start = []
            result_to_goal = []

            t = x
            while t != start:
                result_to_start.append(t)
                t = parent_node_from_start[t]

            t = x
            while t != goal:
                result_to_goal.append(t)
                t = parent_node_from_goal[t]
            return result_to_start.reverse() + result_to_goal
        
        for u in x.successor():
            if not visit_goal[u]:
                parent_node_from_goal[u] = x
                u_cost = x_cost + cost_compute(x, u)
                path_cost_from_goal[u] = u_cost
                heapq.heappush(frontier_from_goal, (u_cost, u))
            else:
                u_cost = x_cost + cost_compute(x, u)
                if u_cost < path_cost_from_goal[u]:
                    parent_node_from_goal[u] = x
                    path_cost_from_goal[u] = u_cost
                    heapq.heappush(frontier_from_goal, (u_cost, u))




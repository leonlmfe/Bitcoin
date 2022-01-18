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


in_path = {}
parent_node = {}

def depth_limit_search(current_node, goal, current_level, limit_level):
    if current_level == limit_level:
        return "Reach Limited!"
    if in_path[current_node]:
        return "Visited!"
    if current_node == goal:
        return "Found!"
    
    in_path[current_node] = True;

    for u in current_node.successor():
        parent_node[u] = current_node
        if depth_limit_search(u, goal, current_level + 1, limit_level) == "Found!":
            return "Found!"
    
    in_path[current_node] = False;

    return "Not Found!"

def search(start, goal):
    for limit_level in range(1, 10000000):
        if depth_limit_search(start, goal, 0, limit_level) == "Found!":
            result = []
            t = goal
            while t != start:
                result.append(t)
                t = parent_node[t]
            return result

 
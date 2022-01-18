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

def ida_star(current_node, goal, current_f, current_g, max_limit):
    if current_f > max_limit:
        return max_limit, False
    if current_node == goal:
        return current_f, True
    
    in_path[current_node] = True;

    for u in current_node.successor():
        if not in_path[u] :
            parent_node[u] = current_node
            u_g = current_g + cost_compute(current_node, u)
            u_h = h(u, goal)
            tmax, flag = ida_star(u, goal, current_level + 1, u_g + u_h, u_g, max_limit)
        
            if flag:
                return tmax, flag
        
            max_limit = min(tmax, max_limit)
    
    in_path[current_node] = False;

    return max_limit, False

def search(start, goal):
    initial_h_value = h(start, goal)
    tmax, flag = ida_star(start, goal, initial_h_value, 0, inf)
    if flag:
        return result
    return "Not FOund"


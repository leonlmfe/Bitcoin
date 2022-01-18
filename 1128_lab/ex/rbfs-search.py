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

def recursive_best_first_search(current_node, goal, current_f, current_g, f_limit):
    if current_node == goal:
        return current_f, True
    
    in_path[current_node] = True;
    
    sg = {}
    sf = {}
    for u in current_node.successor():
        sg[u] = current_g + cost_compute(current_node, u)
        sf[u] = sg[u] + h(u, goal)
    
    while True:
        best_u = second_best_u = None
        best_f = second_best_f = -1

        for u, f in sf.items():
            if best_u is None:
                best_u, best_f = u, f
            elif second_best_u is None:
                if best_f > f:
                    best_u, second_best_u = u, best_u
                    best_f, second_best_f = f, best_f
                else:
                    second_best_u, second_best_f = u, f
            elif best_f > f:
                best_u, second_best_u = u, best_u
                best_f, second_best_f = f, best_f
            elif second_best_f > f:
                second_best_u, second_best_f = u, f
        
        if best_f > f_limit:
            return best_f, False
        
        best_f, result = recursive_best_first_search(best_u, goal, best_f, sg[best_u], min(f_limit, second_best_f))
        sf[best_u] = best_f

        if result:
            return best_f, result
                

def search(start, goal):
    initial_h_value = h(start, goal)
    tmax, flag = ida_star(start, goal, initial_h_value, 0, inf)
    if flag:
        return result
    return "Not FOund"


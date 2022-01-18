
def tree_search(start, goal):
    parent_node = {}
    
    frontier = []
    frontier.append(start)

    while True:
        if not frontier:
            return "Error"
        x = frontier[0]
        frontier = frontier[1:]

        if x == goal:
            result = []
            t = x
            while t != start:
                result.append(t)
                t = parent_node[t]
            return result
        
        for u in x.successor:
            parent_node[u] = x
            frontier.append(u)


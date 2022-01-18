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


visit = {}
parent_node = {}


def depth_first_search(current_node, goal):
    if visit[current_node]:
        return False
    if current_node == goal:
        return True
    
    visit[current_node] = True;

    for u in current_node.successor():
        parent_node[u] = current_node
        if depth_first_search(u, goal) == "Found!":
            return "Found!"

    return "Not Found!"

def search(start, goal):
    if depth_first_search(start, goal) == "Found!":
        result = []
        t = goal
        while t != start:
            result.append(t)
            t = parent_node[t]
        return result

 
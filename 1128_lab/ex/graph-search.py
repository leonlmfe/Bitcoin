
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


def graph_search(start, goal):
    visit = {}
    parent_node = {}
    
    frontier = []
    frontier.append(start)

    while True:
        if not frontier:
            return "Error"
        
        x = frontier[-1]
        visit[x] = True
        del frontier[-1]

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
                frontier.append(u)


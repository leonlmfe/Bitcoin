# 政大資科職碩 110971018 李昂縣
import argparse
import timeit
import resource
from collections import deque
from heapq import heappush, heappop, heapify
import heapq
import itertools
class State:

    def __init__(self, state, parent, move, depth, cost, key):

        self.state = state

        self.parent = parent

        self.move = move

        self.depth = depth

        self.cost = cost

        self.key = key

        if self.state:
            self.map = ''.join(str(e) for e in self.state)

    def __eq__(self, other):
        return self.map == other.map

    def __lt__(self, other):
        return self.map < other.map

goal_state = [5, 8, 6, 0, 7, 4, 2, 3, 1]
goal_node = State
initial_state = list()
board_len = 0
board_side = 0

nodes_expanded = 0
max_search_depth = 0
max_frontier_size = 0

moves = list()
costs = set()
            
    
# Iterative-Deepening Search 
def ids(start_state):

    global costs

    for limit_level in range(1, 32):
        response = dls_mod(start_state, limit_level)

        if type(response) is list:
            return response
            break

        costs = set()


# DLS
def dls_mod(start_state, threshold):

    global max_frontier_size, goal_node, max_search_depth, costs

    explored, stack = set(), list([State(start_state, None, None, 0, 0, threshold)])

    while stack:

        node = stack.pop()

        explored.add(node.map)

        if node.state == goal_state:
            goal_node = node
            return stack

        if node.key > threshold:
            costs.add(node.key)

        if node.depth < threshold:

            neighbors = reversed(expand(node))

            for neighbor in neighbors:
                if neighbor.map not in explored:

                    neighbor.key = neighbor.cost + h(neighbor.state)
                    stack.append(neighbor)
                    explored.add(neighbor.map)

                    if neighbor.depth > max_search_depth:
                        max_search_depth += 1

            if len(stack) > max_frontier_size:
                max_frontier_size = len(stack)

    return min(costs)

# Uniform-Cost Search
def ucs(start_state):
    
    global goal_node
    frontier,explored,path_cost,heap_entry = list(),set(),{},{}
    
    key = h(start_state)
    root = State(start_state, None, None, 0, 0, key)
    entry = (key, 0, root)
    
    heappush(frontier, entry)

    path_cost[root.map] = 0

    heap_entry[root.map] = root
    
    while frontier:
        node = heapq.heappop(frontier)
        explored.add(node[2].map)

        if node[2].state == goal_state:
            goal_node = node[2]
            return frontier

        neighbors = expand(node[2])
        
        for neighbor in neighbors:
# 預設路徑COST皆為1
            u_cost = node[2].cost + 1
            neighbor.key = neighbor.cost + h(neighbor.state)

            entry = (neighbor.key, neighbor.move, neighbor)
            if neighbor.map not in explored:
                path_cost[neighbor.map] = u_cost
                heapq.heappush(frontier, entry)
                explored.add(neighbor.map) 
                heap_entry[neighbor.map] = entry
# 比較路徑cost，若新NODE COST小於原NODE COST 則取代
            elif neighbor.map in heap_entry and u_cost < path_cost[neighbor.map]:
                neighbor.parent = node[2]
                neighbor.cost = u_cost
                path_cost[neighbor.map] = u_cost
                explored.add(neighbor.map)
                hindex = frontier.index((heap_entry[neighbor.map][2].key,
                                     heap_entry[neighbor.map][2].move,
                                     heap_entry[neighbor.map][2]))
                frontier[int(hindex)] = entry
                heap_entry[neighbor.map] = entry
                heapify(frontier)
                
# Greedy Best-First Search
def gbf(start_state):
    global goal_node

    explored, heap, heap_entry = set(), list(), {}

    key = h(start_state)

    root = State(start_state, None, None, 0, 0, key)

    entry = (key, 0, root)

    heappush(heap, entry)

    heap_entry[root.map] = entry

    while heap:

        node = heappop(heap)

        explored.add(node[2].map)

        if node[2].state == goal_state:
            goal_node = node[2]
            return heap

        neighbors = expand(node[2])

        for neighbor in neighbors:

            neighbor.key = h(neighbor.state)

            entry = (neighbor.key, neighbor.move, neighbor)

            if neighbor.map not in explored:

                heappush(heap, entry)

                explored.add(neighbor.map)

                
# A* search
def ast(start_state):

    global goal_node

    explored, heap, heap_entry = set(), list(), {}

    key = h(start_state)

    root = State(start_state, None, None, 0, 0, key)

    entry = (key, 0, root)

    heappush(heap, entry)

    heap_entry[root.map] = entry

    while heap:

        node = heappop(heap)

        explored.add(node[2].map)

        if node[2].state == goal_state:
            goal_node = node[2]
            return heap

        neighbors = expand(node[2])

        for neighbor in neighbors:

            neighbor.key = neighbor.cost + h(neighbor.state)

            entry = (neighbor.key, neighbor.move, neighbor)

            if neighbor.map not in explored:

                heappush(heap, entry)

                explored.add(neighbor.map)

                heap_entry[neighbor.map] = entry

# Recursive Best-First Search
def rbf(start_state):
    global goal_node
    frontier,explored = list(),set()
    
    key = h(start_state)
    root = State(start_state, None, None, 0, 0, key)
    startRate = diffPuzzle(start_state)
    entry = (startRate, 0, root)
    
    heappush(frontier, entry)

    while frontier:
#       heap會自動把[0]從小往大排序  所以會將差異值小的優先處理
        node = heapq.heappop(frontier)
        explored.add(node[2].map)

        if node[2].state == goal_state:
            goal_node = node[2]
            return frontier

        neighbors = expand(node[2])
        
        for neighbor in neighbors:

            n_rate = diffPuzzle(neighbor.state)
            n_node = (n_rate, neighbor.cost, neighbor)
            
            if neighbor.map not in explored:
                heapq.heappush(frontier, n_node)
                explored.add(neighbor.map) 


# 比較方塊差異性，回傳差異方塊數      
def diffPuzzle(current_state):
    
    diffNum = 0
    goal_state
    for i in range(0, 9):
        if goal_state[i] != current_state[i]:
            diffNum = diffNum+1
    return diffNum


def expand(node):

    global nodes_expanded
    nodes_expanded += 1

    neighbors = list()

    neighbors.append(State(move(node.state, 1), node, 1, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 2), node, 2, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 3), node, 3, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 4), node, 4, node.depth + 1, node.cost + 1, 0))

    nodes = [neighbor for neighbor in neighbors if neighbor.state]

    return nodes


def move(state, position):

    new_state = state[:]

    index = new_state.index(0)
    
    if position == 1:  # Up

        if index not in range(0, board_side):

            temp = new_state[index - board_side]
            new_state[index - board_side] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None

    if position == 2:  # Down

        if index not in range(board_len - board_side, board_len):

            temp = new_state[index + board_side]
            new_state[index + board_side] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None

    if position == 3:  # Left

        if index not in range(0, board_len, board_side):

            temp = new_state[index - 1]
            new_state[index - 1] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None

    if position == 4:  # Right

        if index not in range(board_side - 1, board_len, board_side):

            temp = new_state[index + 1]
            new_state[index + 1] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None


def h(state):

    return sum(abs(b % board_side - g % board_side) + abs(b//board_side - g//board_side)
               for b, g in ((state.index(i), goal_state.index(i)) for i in range(1, board_len)))


def backtrace():

    current_node = goal_node
    while initial_state != current_node.state:

        if current_node.move == 1:
            movement = 'Up'
        elif current_node.move == 2:
            movement = 'Down'
        elif current_node.move == 3:
            movement = 'Left'
        else:
            movement = 'Right'

        moves.insert(0, movement)
        current_node = current_node.parent

    return moves


def export(frontier):

    global moves

    moves = backtrace()
#     print("path_to_goal: " + str(moves))
    print("步數: " + str(len(moves))) 
    print("狀態數: " + str(len(frontier)))


def solvable(tiles):
    count = 0

    for i in range(9):
        for j in range(i+1, 9):
            if tiles[j] and tiles[i] and tiles[i] > tiles[j]:
                count += 1

    return count % 2 == 0
    
def read(configuration):

    global board_len, board_side

    data = configuration.split(",")
    if len(data)!=9:
        print("unsolvable")
        return None
    for element in data:
        if element =='':
            print("unsolvable")
            return None
        initial_state.append(int(element)) 
    
    board_len = len(initial_state)
    board_side = int(board_len ** 0.5)

    return "pass"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('algorithm')
    parser.add_argument('board')
    args = parser.parse_args()

    if read(args.board) is None:
        return
    try:
        sum(abs(b % board_side - g % board_side) + abs(b//board_side - g//board_side)
               for b, g in ((initial_state.index(i), goal_state.index(i)) for i in range(1, board_len)))
    except ValueError :
        print("unsolvable")
        return
    function = function_map[args.algorithm]
    if(solvable(initial_state)) :
        print("unsolvable")
        return
    
    frontier = function(initial_state)

    export(frontier)


function_map = {
    'ids': ids,  
    'ucs': ucs,    
    'gbf' : gbf,    
    'ast': ast,
    'rbf' : rbf,
}

if __name__ == '__main__':
    main()
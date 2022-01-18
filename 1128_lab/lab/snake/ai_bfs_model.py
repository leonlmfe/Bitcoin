from game import Model, Direction, Environment
from collections import deque

ROUTE_MARK = -3

class AIBFSModel(Model):
    def __init__(self, logger = None):
        super().__init__(logger)
        self.route = []

    def route_compute(self, board, snake_head, food):
        find = False
        queue = deque([snake_head])

        parent_map = {}

        while queue:
            cur_pos = queue.popleft()
            cx, cy = cur_pos

            for i, (dx, dy) in enumerate(Direction.MOVE):
                tx, ty = cx + dx, cy + dy
                tmp_pos = (tx, ty)
                if board[tx, ty] == Environment.FOOD:
                    parent_map[food] = (cur_pos, i)
                    find = True
                    break
                elif board[tx, ty] == Environment.EMPTY:
                    board[tx, ty] = ROUTE_MARK
                    queue.append(tmp_pos)
                    parent_map[tmp_pos] = (cur_pos, i)

        if find:
            cur_pos = food
            while cur_pos != snake_head:
                cur_pos, d = parent_map[cur_pos]
                self.route.append(d)

        return find

    def move(self, board, snake_head, food):
        if self.route:
            return self.route.pop()
        elif self.route_compute(board, snake_head, food):
            return self.route.pop()
        else:
            return Direction.UP

    def reset(self):
        pass

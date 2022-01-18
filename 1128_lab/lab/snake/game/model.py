
class Model:
    def __init__(self, logger = None):
        self.turn = 0
        self.min_score = self.mean_score = self.max_score = 0
        self.logger = logger

    def user_input(self, event):
        pass
    
    def move(self, board, snake_head, food):
        pass

    def log_score(self, score):
        self.turn += 1

        if self.logger is not None:
            self.logger.log(score)

        if self.turn == 1:
            self.min_score = self.mean_score = self.max_score = score
        else:
            self.min_score = min(self.min_score, score)
            self.mean_score += (score - self.mean_score) / float(self.turn)
            self.max_score = max(self.max_score, score)

    def stats(self):
        return (self.min_score, self.mean_score, self.max_score)

    def reset(self):
        pass

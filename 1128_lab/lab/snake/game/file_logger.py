import io
from .logger import Logger

class FileLogger(Logger):
    def __init__(self):
        self.scores = []
    
    def log(self, score):
        self.scores.append(score)

    def history(self):
        return self.scores

    def save(self, file_name):
        with open(file_name, 'w') as file:
            file.write("\n".join(map(str, self.scores)))

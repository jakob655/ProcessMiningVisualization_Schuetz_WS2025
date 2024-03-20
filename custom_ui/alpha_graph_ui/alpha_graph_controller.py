from api.pickle_save import pickle_load
from mining_algorithms.alpha_mining import AlphaMining


class AlphaGraphController:
    def __init__(self, workingDirectory):
        super().__init__()
        self.model = None
        self.workingDirectory = workingDirectory

    def start_mining(self, cases):
        self.model = AlphaMining(cases)
        self.mine_and_draw()
        pass

    def load_model(self, file_path):
        self.model = pickle_load(file_path)
        self.mine_and_draw()
        return file_path

    def mine_and_draw(self):
        pass

    def get_model(self):
        return self.model

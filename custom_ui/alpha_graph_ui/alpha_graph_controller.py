from api.pickle_save import pickle_load
from mining_algorithms.alpha_mining import AlphaMining


class AlphaGraphController:
    def __init__(self, workingDirectory):
        super().__init__()
        self.model = None
        self.workingDirectory = workingDirectory

    def start_mining(self, cases):
        self.model = AlphaMining(set(cases))
        self.draw_graph()

    def load_model(self, file_path):
        self.model = pickle_load(file_path)
        self.draw_graph()
        return file_path

    def draw_graph(self):
        graph = self.model.draw_graph()
        graph.render(self.workingDirectory, format='dot')
        return graph

    def get_model(self):
        return self.model

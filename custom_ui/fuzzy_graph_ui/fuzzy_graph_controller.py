from mining_algorithms.fuzzy_mining import FuzzyMining
from api.pickle_save import pickle_load


class FuzzyGraphController():
    def __init__(self, workingDirectory, default_significance=0.0, default_correlation=0.5, default_edge_cutoff=0.4,
                 default_utility_ration=0.5):
        super().__init__()
        self.model = None
        self.workingDirectory = workingDirectory
        self.significance = default_significance
        self.edge_cutoff = default_edge_cutoff
        self.utility_ration = default_utility_ration
        self.correlation = default_correlation

    def startMining(self, cases):
        self.model = FuzzyMining(cases)
        self.mine_and_draw(self.significance, self.correlation, self.edge_cutoff, self.utility_ration)

    def mine_and_draw(self, significance, correlation, edge_cutoff, utility_ration):
        graph = self.model.create_graph_with_graphviz(float(significance), float(correlation), float(edge_cutoff),
                                                      float(utility_ration))
        graph.render(self.workingDirectory, format=('dot'))

        return graph

    def loadModel(self, file_path):
        self.model = pickle_load(file_path)
        self.mine_and_draw(self.get_significance(), self.get_correlation(), self.get_edge_cutoff(),
                           self.get_utility_ratio())
        return file_path

    def getModel(self):
        return self.model

    def get_significance(self):
        return self.model.get_significance()

    def get_correlation(self):
        return self.model.get_correlation()

    def get_edge_cutoff(self):
        return self.model.get_edge_cutoff()

    def get_utility_ratio(self):
        return self.model.get_utility_ratio()

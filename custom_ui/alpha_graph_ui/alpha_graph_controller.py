
class AlphaGraphController():
    def __init__(self, workingDirectory, default_significance = 0.0, default_correlation = 0.5, default_edge_cutoff = 0.4, default_utility_ration = 0.5):
        super().__init__()
        self.model = None
        self.workingDirectory = workingDirectory
        self.significance = default_significance
        self.edge_cutoff = default_edge_cutoff
        self.utility_ration = default_utility_ration
        self.correlation = default_correlation
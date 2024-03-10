from graphviz import Digraph
import numpy as np
from mining_algorithms.ddcal_clustering import DensityDistributionClusterAlgorithm

class AlphaMining():
    def __init__(self, case):
        self.case = case
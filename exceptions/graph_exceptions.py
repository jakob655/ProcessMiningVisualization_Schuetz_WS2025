class GraphException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class DuplicateNodeException(GraphException):

    def __init__(self, node):
        self.message = f"Node '{node}' already exists in the graph."
        super().__init__(self.message)


class DuplicateEdgeException(GraphException):

    def __init__(self, source, target):
        self.message = f"Edge from {source} to {target} already exists."
        super().__init__(self.message)


class NodeDoesNotExistException(GraphException):

    def __init__(self, node):
        self.message = f"Node '{node}' does not exist in the graph."
        super().__init__(self.message)


class EdgeDoesNotExistException(GraphException):

    def __init__(self, source, target):
        self.message = f"Edge from {source} to {target} does not exist."
        super().__init__(self.message)


class InvalidNodeNameException(GraphException):

    def __init__(self, node):
        self.message = f"Node '{node}' is not a valid node name."
        super().__init__(self.message)

import unittest
from mining_algorithms.inductive_mining import InductiveMining


def isProcessTreeEqual(tree1, tree2):
    """checks if two process trees are equal

    Parameters
    ----------
    tree1 : tuple(str, tuple(str, ...),...)
        process tree to compare for equality
        the process tree is a tuple where the first element is the operation
        and the rest of the elements are the children of the operation
        the children are also process trees in the same format.
        The leaf nodes are strings

    tree2 : tuple(str, tuple(str, ...),...)
        process tree to compare for equality
        the process tree is a tuple where the first element is the operation
        and the rest of the elements are the children of the operation
        the children are also process trees in the same format.
        The leaf nodes are strings

    Returns
    -------
    bool
        True if the process trees are equal, False otherwise

    Raises
    ------
    Exception
        If the process tree is not in the correct format (tuple, or string, or int leaf nodes)
    """
    if type(tree1) != type(tree2):
        return False

    if isinstance(tree1, str) or isinstance(tree1, int):
        return tree1 == tree2

    if not isinstance(tree1, tuple):
        raise Exception("Invalid tree type")

    if len(tree1) != len(tree2):
        return False

    operation = tree1[0]

    if operation != tree2[0]:
        return False

    # ordered cuts first
    if operation == "seq":
        return all(isProcessTreeEqual(tree1[i], tree2[i]) for i in range(1, len(tree1)))
    if operation == "loop":
        if not isProcessTreeEqual(tree1[1], tree2[1]):
            return False

    for i in range(1, len(tree1)):
        foundEqual = False
        for j in range(1, len(tree2)):
            if isProcessTreeEqual(tree1[i], tree2[j]):
                foundEqual = True
                break

        if not foundEqual:
            return False

    return True


class TestInductiveMiner(unittest.TestCase):

    def test_inductive_mining_with_only_one_cut(self):
        log = {(1, 2, 3): 6}

        inductive_mining = InductiveMining(log)
        result = inductive_mining.inductive_mining(log)

        self.assertTrue(isProcessTreeEqual(result, ("seq", 1, 2, 3)))

    def test_inductive_miner_with_multiple_cuts(self):
        log = {
            (1, 2, 3, 4): 2,
            (1, 3, 2, 4): 5,
            (1, 2, 3, 5, 6, 2, 3, 4): 3,
            (1, 3, 2, 5, 6, 3, 2, 4): 1,
        }

        inductive_mining = InductiveMining(log)
        result = inductive_mining.inductive_mining(log)

        expected_tree = ("seq", 1, ("loop", ("par", 2, 3), ("seq", 5, 6)), 4)

        self.assertTrue(isProcessTreeEqual(result, expected_tree))

    def test_inductive_miner_fallthrough_with_empty_sublog(self):
        log = {(1, 2, 3): 2, (1, 3): 1, (): 1}

        inductive_mining = InductiveMining(log)
        result = inductive_mining.inductive_mining(log)

        expected_tree = ("xor", "tau", ("seq", 1, ("xor", "tau", 2), 3))

        self.assertTrue(isProcessTreeEqual(result, expected_tree))

    def test_fallthrough_with_one_event_more_than_once_in_trace(self):
        log = {(1,): 1, (1, 1, 1): 5, (1, 1): 1}

        inductive_mining = InductiveMining(log)
        result = inductive_mining.inductive_mining(log)

        self.assertTrue(isProcessTreeEqual(result, ("loop", 1, "tau")))

    def test_flower_model_fallthrough(self):
        log = {(1, 2, 3): 1, (2, 3, 1): 1}

        inductive_mining = InductiveMining(log)
        result = inductive_mining.inductive_mining(log)

        expected_tree = ("loop", "tau", 1, 2, 3)

        self.assertTrue(isProcessTreeEqual(result, expected_tree))

    def test_inductive_miner_with_test_csv(self):
        log = {
            ("a", "e"): 5,
            ("a", "b", "c", "e"): 10,
            ("a", "c", "b", "e"): 10,
            ("a", "b", "e"): 1,
            ("a", "c", "e"): 1,
            ("a", "d", "e"): 10,
            ("a", "d", "d", "e"): 2,
            ("a", "d", "d", "d", "e"): 1,
        }

        inductive_mining = InductiveMining(log)
        result = inductive_mining.inductive_mining(log)

        expected_tree = (
            "seq",
            "a",
            (
                "xor",
                "tau",
                (
                    "xor",
                    ("loop", "d", "tau"),
                    ("par", ("xor", "tau", "b"), ("xor", "tau", "c")),
                ),
            ),
            "e",
        )
        self.assertTrue(isProcessTreeEqual(result, expected_tree))


if __name__ == "__main__":
    unittest.main()

from graphs.dfg import DFG
from collections import deque


def exclusive_cut(graph: DFG) -> list[set[str | int]]:
    """Calculate the exclusive cut of a graph. The exclusive cut is a partitioning of the graph into multiple connected components.
    If only one connected component exists, the exclusive cut is not possible and None is returned.

    Parameters
    ----------
    graph : DFG
        The graph that should be partitioned

    Returns
    -------
    list[set[str | int]]
        A list of connected components that represent the exclusive cut of the graph
    """
    # check if the graph has only one start node or one end node
    # if this is the case return None, as no exclusive cut is possible
    if len(graph.get_start_nodes()) == 1 or len(graph.get_end_nodes()) == 1:
        return None

    connected_components = graph.get_connected_components()
    return connected_components if len(connected_components) > 1 else None


def sequence_cut(graph: DFG) -> list[set[str | int]]:
    """Calculate the sequence cut of a graph. The sequence cut is an ordered cut,
    where each a only a previous partition can reach a following partition, but not the other way around.
    This implementation uses a greedy algorithm to merge partitions that are reachable or not reachable from each other.
    After the merging, the partitions are sorted by the order of reachability.
    If only one partition exists, the sequence cut is not possible and None is returned.

    Parameters
    ----------
    graph : DFG
        The graph that should be partitioned

    Returns
    -------
    list[set[str | int]]
        A list of partitions that represent the sequence cut of the graph
    """
    partitions = [{node} for node in graph.get_nodes()]
    nodes = list(graph.get_nodes())

    reachable_nodes = {node: graph.get_reachable_nodes(node) for node in nodes}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_1 = nodes[i]
            node_2 = nodes[j]
            is_j_reachable_from_i = node_2 in reachable_nodes[node_1]
            is_i_reachable_from_j = node_1 in reachable_nodes[node_2]

            # merge partitions if the nodes are reachable from each other or not reachable from each other
            if (is_j_reachable_from_i and is_i_reachable_from_j) or (
                not is_i_reachable_from_j and not is_j_reachable_from_i
            ):
                partition_1 = None
                partition_2 = None

                for partition in partitions:
                    if partition_1 and partition_2:
                        break

                    if node_1 in partition:
                        partition_1 = partition

                    if node_2 in partition:
                        partition_2 = partition

                if partition_1 != partition_2:
                    partition_1.update(partition_2)
                    partitions.remove(partition_2)

    # sort partitions by the order of reachability i < j if i is reachable from j
    for i in range(len(partitions)):
        min_partition_index = i
        for j in range(i + 1, len(partitions)):
            if (
                next(iter(partitions[min_partition_index]))
                in reachable_nodes[next(iter(partitions[j]))]
            ):
                min_partition_index = j

        partitions[i], partitions[min_partition_index] = (
            partitions[min_partition_index],
            partitions[i],
        )

    return partitions if len(partitions) > 1 else None


def parallel_cut(graph: DFG) -> list[set[str | int]]:
    """Calculate the parallel cut of a graph. The parallel cut is found by first removing all double edges from the graph
    and then adding double edges between nodes with only one edge or no edge between them.
    The the conntected components of the resulting graph are the partitions of the parallel cut.
    Each partition has to contain at least one start node or one end node.
    If only one connected component exists, the parallel cut is not possible and None is returned.


    Parameters
    ----------
    graph : DFG
        The graph that should be partitioned

    Returns
    -------
    list[set[str | int]]
        A list of partitions that represent the parallel cut of the graph
    """
    # check if the graph has only one start node or one end node
    # if this is the case return None, as no exclusive cut is possible
    if len(graph.get_start_nodes()) == 1 or len(graph.get_end_nodes()) == 1:
        return None

    inverted_dfg = graph.invert()
    partitions = inverted_dfg.get_connected_components()

    if len(partitions) == 1:
        return None

    partitions_to_merge = [
        partition
        for partition in partitions
        if not partition.intersection(graph.get_start_nodes())
        or not partition.intersection(graph.get_end_nodes())
    ]

    partitions = [
        partition for partition in partitions if partition not in partitions_to_merge
    ]

    partitions_with_only_start_nodes = list()
    partitions_with_only_end_nodes = list()
    partitions_without_start_end_nodes = list()

    # split partitions into partitions with only start nodes, partitions with only end nodes and partitions without start and end nodes
    for partition in partitions_to_merge:
        has_start_node = len(partition.intersection(graph.get_start_nodes())) > 0
        has_end_node = len(partition.intersection(graph.get_end_nodes())) > 0

        if has_start_node:
            partitions_with_only_start_nodes.append(partition)

        elif has_end_node:
            partitions_with_only_end_nodes.append(partition)

        else:
            partitions_without_start_end_nodes.append(partition)

    partitions_to_merge = partitions_without_start_end_nodes

    # merge partitions with only start nodes with partitions with only end nodes
    for start_node_partition, end_node_partition in zip(
        partitions_with_only_start_nodes, partitions_with_only_end_nodes
    ):
        partitions.append(start_node_partition.union(end_node_partition))

    # if the number of partitions with only start nodes is not equal to the number of partitions with only end nodes
    # merge the remaining partitions with the other partitions to merge
    if len(partitions_with_only_end_nodes) != len(partitions_with_only_start_nodes):
        index = min(
            len(partitions_with_only_start_nodes), len(partitions_with_only_end_nodes)
        )

        if len(partitions_with_only_start_nodes) > len(partitions_with_only_end_nodes):
            partitions_to_merge.extend(partitions_with_only_start_nodes[index:])
        else:
            partitions_to_merge.extend(partitions_with_only_end_nodes[index:])

    # merge partitions without start and end nodes with partitions with start and end nodes
    index_to_merge = 0
    for partition in partitions_to_merge:
        partitions[index_to_merge] = partitions[index_to_merge].union(partition)
        index_to_merge = (index_to_merge + 1) % len(partitions)

    return partitions if len(partitions) > 1 else None


def loop_cut(graph: DFG) -> list[set[str | int]]:
    """Calculate the loop cut of a graph. The loop cut is a partly ordered cut,
    where the first partitions represents the do part of the loop and the second partition the redo part of the loop.
    The first partitions contains all start and end nodes. The other partitions are found by first removing all start and end nodes from the graph.
    Then the connected components of the resulting graph are the partitions of the loop cut.

    Then the following conditions are checked for each partition that is not the first partition:
    1. There is no edge from a start node, that is not also a end node, to a node in the partition
    2. There is no edge from a node in the partition to an end node, that is not also a start node
    3. if there is an edge from an end node to a node in the connected component there have to be edges from all end nodes to this node
    4. if there is an edge from a node in the connected component to a start node there have to be edges from this node to all start nodes

    If one of the conditions is not fullfiled the partition is merged with the first partition.
    If only one partition exists, the loop cut is not possible and None is returned.

    Parameters
    ----------
    graph : DFG
        The graph that should be partitioned

    Returns
    -------
    list[set[str | int]]
        A list of partitions that represent the loop cut of the graph
    """
    partition_1 = set(graph.get_start_nodes().union(graph.get_end_nodes()))

    starting_nodes = graph.get_start_nodes()
    ending_nodes = graph.get_end_nodes()

    dfg_without_end_start_nodes = graph.create_dfg_without_nodes(
        starting_nodes.union(ending_nodes)
    )

    # create partitions thath are not connected to each other
    connected_components = dfg_without_end_start_nodes.get_connected_components()

    only_start_nodes = set(graph.get_start_nodes() - graph.get_end_nodes())
    only_end_nodes = set(graph.get_end_nodes() - graph.get_start_nodes())

    # check other conditions for partitions if not fullfiled merge with partition_1

    # check if there is a conection from a start node (that is not an end node) to a node in the connected component
    # if there is such a edge, there partition is merged with partition_1

    filtered_partitions = list()
    for partition in connected_components:
        partition_merged = False
        for node in partition:
            for start_node in only_start_nodes:
                if graph.contains_edge(start_node, node):
                    partition_1.update(partition)
                    partition_merged = True
                    break

            if partition_merged:
                break
        if not partition_merged:
            filtered_partitions.append(partition)

    connected_components = filtered_partitions
    filtered_partitions = list()

    # check if there is a conection from a node in the connected component to an end node (that is not a start node)
    # if there is such a edge, there partition is merged with partition_1

    for partition in connected_components:
        partition_merged = False
        for node in partition:
            for end_node in only_end_nodes:
                if graph.contains_edge(node, end_node):
                    partition_1.update(partition)
                    partition_merged = True
                    break

            if partition_merged:
                break
        if not partition_merged:
            filtered_partitions.append(partition)

    connected_components = filtered_partitions
    filtered_partitions = list()

    # check for all connected components the following conditions
    # 1. if there is an edge from an end node to a node in the connected component there have to be edges from all end nodes to this node
    # 2. if there is an edge from a node in the connected component to a start node there have to be edges from this node to all start nodes
    # if one of the conditions is not fullfiled merge the connected component with partition_1

    for partition in connected_components:
        merged_partition = False
        for node in partition:
            for end_node in ending_nodes:
                if graph.contains_edge(end_node, node):
                    for other_end_node in ending_nodes:
                        if not graph.contains_edge(other_end_node, node):
                            partition_1.update(partition)
                            merged_partition = True
                            break
                if merged_partition:
                    break
            if merged_partition:
                break
            for start_node in starting_nodes:
                if graph.contains_edge(node, start_node):
                    for other_start_node in starting_nodes:
                        if not graph.contains_edge(node, other_start_node):
                            partition_1.update(partition)
                            merged_partition = True
                            break

                if merged_partition:
                    break
        if not merged_partition:
            filtered_partitions.append(partition)

    partitions = [partition_1, *filtered_partitions]
    return partitions if len(partitions) > 1 else None

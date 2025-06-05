# Heuristic Miner

The heuristic miner is a simple, and effective process mining algorithm. It considers the frequencies of events and directly-follows relations during the creation of the process model, and focuses on the most common paths in the process while removing infrequent ones. Different metrics are used to remove connections and events from the process model.

The algorithm creates a directly-follows graph and stores it as a succession matrix. The different metrics are used to find infrequent nodes and edges, which are then discarded.

## Metrics

Currently, three metrics are used to simplify the graph, the **spm metric**, the **frequency metric** and the **dependency metric**.

The **SPM filter** simplifies process models by removing low-quality nodes based on their frequency and connectivity. It scores each node using the **spm metric**, which balances complexity and common behavior. This abstraction helps generate clearer, more interpretable models. Especially useful for user-driven processes like search behavior.

The **frequency metric** is calculated for edges and nodes/events. The event frequency counts the occurrences of an event in the log. The edge frequency counts the number of times one event is directly followed by another event.

The **dependency metric** determines, how one-sided a relationship is between two edges. It compares how dependent an edge is by the following formula:

$$
\text{if a} \neq \text{b} :   D(a > b) = \frac{S(a > b) - S(b > a)}{S(a > b) + S(b > a) + 1}\\

\text{if a} = \text{b} :   D(a > a) = \frac{S(a > a)}{S(a > a) + 1}
$$

where **S(a > b)** means the entry in the succession matrix from a to b

## Filtering

There are three filtering parameters in the current implementation of the heuristic miner: the **SPM filter**, **minimum frequency**, and the **dependency threshold**.

The **SPM value** ranges from 0.0 to 1.0 and reflects the semantic quality of a node. It defines the threshold below which nodes are considered low-quality due to low frequency or high complexity, and may be removed for abstraction.

The **minimum frequency** filters out edges, that have a lower frequency than that threshold. The range of this threshold is from 1 to the maximum edge frequency. Additionally, nodes with a lower frequency are also removed.

The **dependency threshold** is in the range of 0.0 to 1.0. It removed edges, that have a lower dependency score than that threshold.
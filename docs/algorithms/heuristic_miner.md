# Heuristic Miner

The heuristic miner is a simple, and effective process mining algorithm. It considers the frequencies of events and directly-follows relations during the creation of the process model, and focuses on the most common paths in the process while removing infrequent ones. Different metrics are used to remove connections and events from the process model.

The algorithm creates a directly-follows graph and stores it as a succession matrix. The different metrics are used to find infrequent nodes and edges, which are then discarded.

## Metrics

Currently, four metrics are used to simplify the graph, the **SPM metric**, the **Node frequency metric**, the **Edge frequency metric** and the **dependency metric**.

The **SPM filter** simplifies process models by removing low-quality nodes based on their frequency and connectivity. It scores each node using the **spm metric**, which balances complexity and common behavior. This abstraction helps generate clearer, more interpretable models. Especially useful for user-driven processes like search behavior.

*For more information on the SPM metric, see: Marian Lux, Stefanie Rinderle-Ma, Andrei Preda: “Assessing the Quality of Search Process Models.”  
Available online: https://ucrisportal.univie.ac.at/en/publications/assessing-the-quality-of-search-process-models*

**Node frequency (normalized)**  
Measures how often each event (node) appears relative to the most frequent node in the (SPM-filtered) log.  
The result is a value between 0.0 and 1.0. Nodes with a normalized frequency below the threshold are removed.  

**Node frequency (absolute)**  
Counts the total number of occurrences of each event (node) in the log. Nodes with an absolute frequency below the threshold are removed.

**Edge frequency (normalized)**  
Reflects how often a direct transition (edge) between two nodes occurs relative to the most frequent edge in the log.  
The result is a value between 0.0 and 1.0. Edges with a normalized frequency below the threshold are removed.  

**Edge frequency (absolute)**  
Counts the total number of times one event is directly followed by another in the log.  
Edges with an absolute frequency below the threshold are removed.

*The node and edge thresholds are synchronized: adjusting one automatically updates the other, so the normalized and absolute values always stay consistent.*

The **dependency metric** determines, how one-sided a relationship is between two edges. It compares how dependent an edge is by the following formula:

$$
\text{if a} \neq \text{b} :   D(a > b) = \frac{S(a > b) - S(b > a)}{S(a > b) + S(b > a) + 1}\\

\text{if a} = \text{b} :   D(a > a) = \frac{S(a > a)}{S(a > a) + 1}
$$

where **S(a > b)** means the entry in the succession matrix from a to b

## Filtering

There are four filtering parameters in the current implementation of the heuristic miner: the **SPM filter**, **node frequency**, **edge frequency** and the **dependency threshold**.

The **SPM value** ranges from 0.0 to 1.0 and reflects the semantic quality of a node. It defines the threshold below which nodes are considered low-quality due to low frequency or high complexity, and may be removed for abstraction.

The **node frequency (normalized)** value is in the range of 0.0 to 1.0. It filters out nodes that appear too rarely in the log, relative to the most frequent node.  
The **node frequency (absolute)** value is an integer count of event occurrences in the log. It filters out nodes below a chosen absolute frequency.

The **edge frequency (normalized)** value is in the range of 0.0 to 1.0. It filters out transitions (edges) that occur too infrequently in the log, relative to the most frequent edge.  
The **edge frequency (absolute)** value counts the raw number of times one event is directly followed by another. Edges below this threshold are removed.

The **dependency threshold** is in the range of 0.0 to 1.0. It removed edges, that have a lower dependency score than that threshold.
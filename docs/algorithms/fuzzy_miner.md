# Fuzzy Miner

The fuzzy miner is a mining algorithm, that is best used for less structured processes. The algorithm uses the significance and the correlation metric to abstract the process model. Nodes can be collapsed to clusters and edges can be merged for an easier view of the process model.

The algorithm follows three rules. Significant nodes are kept and will not be merged, less significant nodes and highly correlated nodes will be merged into a cluster, and less significant and lowly correlated nodes will be removed. Edge filtering uses the edge cutoff metric, that will be described later.

## Metrics

The fuzzy miner uses the following metrics: **SPM metric**, **Node frequency**, **Edge frequency**, **Unary Significance**, **Binary Significance**, **Correlation**, **Utility Ratio**, **Utility Value**, and **Edge Cutoff**.

The **SPM filter** simplifies process models by removing low-quality nodes based on their frequency and connectivity. It scores each node using the SPM metric, which balances complexity and common behavior. This abstraction helps generate clearer, more interpretable models, especially useful for user-driven processes like search behavior.

*For more information on the SPM metric, see: Marian Lux, Stefanie Rinderle-Ma, Andrei Preda: “Assessing the Quality of Search Process Models.”  
Available online: https://ucrisportal.univie.ac.at/en/publications/assessing-the-quality-of-search-process-models*

**Node frequency** measures how often each event (node) appears relative to the most frequent node in the SPM-filtered log. The value is normalized to fall between 0.0 and 1.0 and is primarily used for filtering nodes that appear too infrequently to be considered relevant.

**Edge frequency** reflects how often a direct transition (edge) between two nodes occurs, again normalized by the maximum edge frequency in the log. This value is used to filter out weak or noisy transitions.

**Unary significance** also reflects how frequently a node appears, using the same formula and normalization as node frequency:

`unary_significance(event) = frequency(event) / max(frequency of all events)`

This value lies between 0.0 and 1.0 and reflects how central a node is within the process.

Although node frequency and unary significance are numerically the same, they serve different purposes:
- *Node frequency* is used for filtering, to directly remove infrequent nodes after SPM filtering.
- *Unary significance*  is used for clustering and abstraction, influencing how nodes are grouped or visually emphasized in the model.

Even though the values are the same, they serve different purposes. And importantly, nodes with high unary significance can still be removed if they are weakly connected (low correlation) to the rest of the process. This underlines how correlation plays a critical role in shaping the final structure.

**Binary significance**, also referred to as **edge significance**, quantifies the importance of a transition (edge) between two events. It combines both the source node’s frequency and the frequency of that edge. In this implementation, binary significance is part of the **utility value**, which is used to filter edges. Find more below at **Utility value**.

If the binary significance of an edge is below the binary significance threshold, the edge is ignored in the utility calculation and may be removed.

**Correlation** measures how strongly two events are related. It is calculated by dividing the number of times an edge between two events occurs by the total number of outgoing edges from the source node:

`correlation(A, B) = frequency(A → B) / sum(frequency(A → *))`

**Utility ratio** defines the weighting between binary significance and correlation when computing edge utility. A value of 1.0 considers only significance; a value of 0.0 considers only correlation.

**Utility value** is the weighted combination of correlation and significance as defined above.

`utility(A, B) = utility_ratio * significance(A, B) + (1 - utility_ratio) * correlation(A, B)`

**Edge cutoff** is calculated by applying Min-Max normalization on the utility values of all incoming edges to a target node:

`normalized_utility = (utility - min) / (max - min)`

Edges with normalized utility below the cutoff value are removed.


## Filtering

Eight metrics are used for filtering in the fuzzy miner algorithm: **spm threshold**, **node frequency*, **edge frequency**, **unary significance**, **binary significance**, **correlation**, **edge cutoff**, and **utility ratio**.

The **SPM** value ranges from 0.0 to 1.0 and reflects the semantic quality of a node. It defines the threshold below which nodes are considered low-quality due to low frequency or high complexity, and may be removed for abstraction.

The **node frequency** value is in the range of 0.0 to 1.0. It filters out nodes that appear too rarely in the log. Nodes below this threshold are removed after the SPM filtering step.

The **edge frequency** value is in the range of 0.0 to 1.0. It filters out transitions (edges) that occur too infrequently in the log. These edges are removed before utility-based edge filtering is applied.

The **unary significance** value is in the range of 0.0 to 1.0. It defines the threshold until a node is considered frequent and relevant. Nodes below this value may be removed or clustered.

The **binary significance** value is in the range of 0.0 to 1.0. It filters out transitions (edges) whose source node is not sufficiently significant. Edges below this value are ignored in the utility calculation.

The **correlation** value is in the range of 0.0 to 1.0. It defines the threshold until transitions between two nodes are considered weakly related and may be removed.

The **edge cutoff** value is also in the range of 0.0 to 1.0. It removes all edges whose normalized utility is below this threshold, simplifying the process model more aggressively.

The **utility ratio** is a configurable value between 0.0 and 1.0. It determines the balance between binary significance and correlation in calculating edge utility. Higher values favor significance, lower values favor correlation.
# Fuzzy Miner

The fuzzy miner is a mining algorithm, that is best used for less structured processes. The algorithm uses the significance and the correlation metric to abstract the process model. Nodes can be collapsed to clusters and edges can be merged for an easier view of the process model.

The algorithm follows three rules. Significant nodes are kept and will not be merged, less significant nodes and highly correlated nodes will be merged into a cluster, and less significant and lowly correlated nodes will be removed. Edge filtering uses the edge cutoff metric, that will be described later.

## Metrics

The fuzzy miner uses the following metrics, spm metric, unary significance, binary significance, correlation, utility ratio, utility value and the edge cutoff value.

The SPM filter simplifies process models by removing low-quality nodes based on their frequency and connectivity. It scores each node using the spm metric, which balances complexity and common behavior. This abstraction helps generate clearer, more interpretable models—especially useful for user-driven processes like search behavior.

The unary significance describes the relative importance of an event. A frequency significance is used, that counts the frequency of all the events and divides it by the maximum event frequency.

The binary significance, or edge significance, is calculated by taking the source node's significance.

Correlation measures how closely related two events are. All edges between two nodes are counted, and divided by the sum of all edges with the same source.

The utility ratio defines a ratio to calculate the utility value by weighting the correlation and significance of an edge.

The utility value is defined by using the binary significance, correlation, and the utility ratio.

`util (A,B) = utility_ratio *significance(A,B) + (1 - utility_ratio)* correlation(A,B)`

The edge cutoff is calculated by normalizing the utility value for edges with the same target node. The Min-Max normalization is used.

## Filtering

Six metrics are used for filtering in the fuzzy miner algorithm: **spm threshold** **unary significance**, **binary significance**, **correlation**, **edge cutoff**, and **utility ratio**.

The **SPM** value ranges from 0.0 to 1.0 and reflects the semantic quality of a node. It defines the threshold below which nodes are considered low-quality due to low frequency or high complexity, and may be removed for abstraction.

The **unary significance** value is in the range of 0.0 to 1.0. It defines the threshold until a node is considered frequent and relevant. Nodes below this value may be removed or clustered.

The **binary significance** value is in the range of 0.0 to 1.0. It filters out transitions (edges) whose source node is not sufficiently significant. Edges below this value are ignored in the utility calculation.

The **correlation** value is in the range of 0.0 to 1.0. It defines the threshold until transitions between two nodes are considered weakly related and may be removed.

The **edge cutoff** value is also in the range of 0.0 to 1.0. It removes all edges whose normalized utility is below this threshold, simplifying the process model more aggressively.

The **utility ratio** is a configurable value between 0.0 and 1.0. It determines the balance between binary significance and correlation in calculating edge utility. Higher values favor significance, lower values favor correlation.
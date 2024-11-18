# Fuzzy Miner

The fuzzy miner is a mining algorithm, that is best used for less structured processes. The algorithm uses the significance and the correlation metric to abstract the process model. Nodes can be collapsed to clusters and edges can be merged for an easier view of the process model.

The algorithm follows three rules. Significant nodes are kept and will not be merged, less significant nodes and highly correlated nodes will be merged into a cluster, and less significant and lowly correlated nodes will be removed. Edge filtering uses the edge cutoff metric, that will be described later.

## Metrics

The fuzzy miner uses the following metrics, unary significance, binary significance, correlation, utility ratio, utility value and the edge cutoff value.

The unary significance describes the relative importance of an event. A frequency significance is used, that counts the frequency of all the events and divides it by the maximum event frequency.

The binary significance, or edge significance, is calculated by taking the source node's significance.

Correlation measures how closely related two events are. All edges between two nodes are counted, and divided by the sum of all edges with the same source.

The utility ratio defines a ratio to calculate the utility value by weighting the correlation and significance of an edge.

The utility value is defined by using the binary significance, correlation, and the utility ratio.

util (A,B) = utility_ratio *significance(A,B) + (1 - utility_ratio)* correlation(A,B)

The edge cutoff is calculated by normalizing the utility value for edges with the same target node. The Min-Max normalization is used.

## Filtering

Three metrics are used for filtering, the significance, the correlation and the edge cutoff.

The significance value is in the range of 0.0 and 1.0. This value defines the threshold until a node is considered significant.

The correlation value has the same range and defines the threshold until nodes are considered lowly correlated.

The edge cutoff value is also in a range of 0.0 and 1.0. It removes all edges that have a lower edge cutoff value than this threshold.

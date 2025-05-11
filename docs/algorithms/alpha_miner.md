# Alpha Miner

The Alpha Miner is a fundamental process mining algorithm that constructs a process model by analyzing the behavior recorded in event logs. It is based on the idea of discovering causal relations, parallel behavior, and choices in the process.

The algorithm is particularly suitable for well-structured processes, as it assumes that the process is fully observable and recorded without noise. It produces a workflow net that models the underlying process.

---

## Algorithm Overview

The Alpha Miner works by analyzing traces (sequences of events) in the event log. It identifies relationships between events and constructs a process graph using the following steps:

1. **Identify Unique Events**:
   
   - Determine all distinct events present in the log.

2. **Find Start and End Events**:
   
   - Identify the first event in each trace (start events) and the last event in each trace (end events).

3. **Discover Relations**:
   
   - Detect relationships between events:
     - **Direct succession**: Event A directly follows Event B.
     - **Causality**: Event A leads to Event B but not vice versa.
     - **Parallelism**: Event A and Event B can occur in any order.
     - **Choice**: Event A and Event B represent alternative paths.

4. **Generate Sets**:
   
   - Create sets of activities that form the basis for the process model.

5. **Construct the Process Graph**:
   
   - Translate the relations and sets into a Petri net or process graph using nodes and edges.

---

## Key Features

- **Causality Analysis**: Identifies the flow of activities in the process by analyzing causal dependencies between events.
- **Parallelism**: Recognizes activities that can occur simultaneously.
- **Choice Discovery**: Detects alternative paths in the process.
- **Graph Representation**: Produces a clear graphical representation of the process, such as a Petri net.

---

## Use Cases

- Best suited for structured processes with clear traces.
- Ideal for processes where all events are recorded without noise or deviations.
- Useful in educational contexts for understanding the basics of process mining.

---

## Implementation Details

1. **Inputs**:
   
   - **Event Log**: A collection of traces, where each trace is a sequence of events.

2. **Outputs**:
   
   - **Process Graph**: A Petri net or similar graph structure representing the process model.

3. **Steps**:
   
   - Extract relationships like direct succession, causality, parallelism, and choice.
   - Formulate sets for transitions and activities.
   - Draw the process graph using a visualization tool like Graphviz.

---

## Metrics/Filtering
### SPM Filter (Search Process Quality Metric)
The SPM filter simplifies process models by removing low-quality nodes based on their frequency and connectivity. It scores each node using the spm metric, which balances complexity and common behavior. This abstraction helps generate clearer, more interpretable models which is especially useful for user-driven processes like search behavior.

The SPM value ranges from 0.0 to 1.0 and reflects the semantic quality of a node. It defines the threshold below which nodes are considered low-quality due to low frequency or high complexity, and may be removed for abstraction.

---

## Limitations

- **Assumes Perfect Data**: The algorithm does not handle noise or incomplete data effectively.
- **Structured Processes Only**: Less effective for unstructured or flexible processes.
- **Scalability**: May struggle with very large event logs due to computational complexity.

---

## References

- The methodology and algorithm were introduced by **Professor Wil van der Aalst** in his lectures on process mining.
- A detailed explanation can be found in the video: [Introduction to Alpha Miner](https://www.youtube.com/watch?v=ATBEEEDxHTQ).

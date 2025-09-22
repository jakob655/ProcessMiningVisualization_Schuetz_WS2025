# Genetic Miner

The Genetic Miner is a global process discovery algorithm based on **genetic algorithms (GAs)**.  
It was introduced by Alves de Medeiros, Weijters, and van der Aalst (2005) as one of the first process mining approaches explicitly designed to handle **noise** and **incompleteness** in event logs [Medeiros, A & Weijters, A. & Aalst, Wil. (2005)](https://www.researchgate.net/publication/228881627_Using_genetic_algorithms_to_mine_process_models_representation_operators_and_results).

Unlike local search-based miners (e.g., Alpha, Heuristic, Inductive), the Genetic Miner treats process discovery as a **global search problem** over the space of possible models. Candidate models (individuals) are evolved through GA operators (elitism, selection, crossover, mutation), guided by a fitness measure that compares model behavior with observed traces.

---

## Algorithm Overview

1. ### **Initialization**
- **Individual Representation**  
  Each individual encodes a candidate process model with:
    - `activities`: all activity labels from the log.
    - `I`: INPUT sets - for each activity, a list of subsets of predecessors.  
      - Each subset = OR-relation (any one is sufficient).  
      - The list of subsets = AND-relation (all subsets must be satisfied).
    - `O`: OUTPUT sets - for each activity, a list of subsets of successors.  
    - `C`: causal relation pairs `(a, b)` derived from the OUTPUT sets.
- **Dependency Measures**  
  Computed from the log with virtual `start` and `end` events:
    - `follows(a, b)`: number of times `a` is immediately followed by `b`.  
    - `L1L(a)`: count of self-loops (`a -> a`).  
    - `L2L(a, b)`: count of length-two loops (`a -> b -> a`).  
    - From these counts:
      - Dependency measure `D(a, b)` is computed using three cases (length-two loop, normal pair, or self-loop).  
      - Start measure `S(t) = D(start, t)`.  
      - End measure `E(t) = D(t, end)`.
- **Heuristic Creation of Individuals**  
  - Step 1: build probabilistic causal matrix - relation `(a, b)` is kept with probability `D(a, b)^power_value`.  
  - Step 2: apply start wipe - with probability `S(a)^power_value`, remove all incoming connections to `a`.  
  - Step 3: apply end wipe - with probability `E(a)^power_value`, remove all outgoing connections from `a`.  
  - Step 4: build INPUT sets by partitioning predecessors of each activity.  
  - Step 5: build OUTPUT sets by partitioning successors.  
  - Step 6: derive causal relation set `C` from OUTPUT sets.
- **Power Parameter (`power_value`)**  
  Controls sparsity of causal relations.  
  - `p=1`: dense initialization (many relations kept).  
  - Higher odd values (e.g. 5, 7, 9): sparse initialization (only strong relations survive).

2. ### **Genetic Evolution**
- The population of individuals is evolved across generations, using GA operators:
  - **Fitness Evaluation**  
    - Each trace is parsed with a token-game simulation.
    - An event is considered parsed if it successfully fires (all of its input conditions are satisfied).
    - Tokens are consumed from INPUT sets and produced into OUTPUT sets.
    - INPUT semantics: an activity is enabled if all of its input sets are satisfied  
      * A single input set is satisfied if at least one of its members has a token (OR).  
      * All input sets must be satisfied together (AND across sets).  
    - Self-loops are handled explicitly.
    - A trace is complete if no tokens remain at the end.  
    - Fitness function:

  $$
  Fitness(PM, L) = 0.40 \cdot \frac{\text{parsed activities}}{\text{total activities}} + 0.60 \cdot \frac{\text{completed traces}}{\text{total traces}}
  $$

  - **Elitism**  
    - The top `elitism_rate × population_size` individuals are copied unchanged into the next generation.
  - **Tournament Selection**  
    - Randomly select `tournament_size` individuals.  
    - The fittest among them becomes a parent.
  - **Crossover**  
    - Randomly select an activity `t`.  
    - Swap suffixes of INPUT partitions between two parents.  
    - Swap suffixes of OUTPUT partitions.  
    - Resolve overlaps by merging/splitting subsets.  
    - Update I/O consistency (`b in O(a) <=> a in I(b)`).
  - **Mutation**  
    - For each activity, with probability `mutation_rate`:  
      - Merge two subsets of INPUT sets or split one subset into two.  
      - Merge/split OUTPUT sets analogously.  
    - After mutation, enforce consistency between INPUT and OUTPUT sets.
  - **Generation Loop**  
    - Start with elite individuals.  
    - Fill the rest of the population with children produced by crossover and mutation.
    - Track best fitness.

3. ### **Termination**
- The genetic algorithm stops when one of the following conditions is met:
  - **Fitness threshold (configurable):**  
    - Immediate stop as soon as any individual reaches `fitness ≥ fitness_threshold` (checked already in generation 0 and during offspring creation).
    - **Optimal fitness (default)**: `fitness = 1.00`
  - **Generation limit**:  
    - If the number of iterations reaches `max_generations`, the algorithm terminates regardless of fitness.
  - **Stagnation**:  
    - If the best fitness score does not improve for `max_generations / 2` consecutive generations, the run is stopped early to prevent wasted computation.

- **Best Individual Tracking**  
  - At the end of the run (or upon early termination), the best individual found so far is stored in `self.best_individual`.  
  - This ensures reproducibility of results and graph construction from the strongest candidate.

4. ### **Graph Construction**
- After termination, the fittest individual is translated into a **Petri-net–like graph** for visualization.
  - **Steps**:
    - **Start and End Nodes**  
      - `Start` and `End` nodes are added.  
      - `Start activities` connect via an intermediate place `p_start`.
      - `End activities` connect via an intermediate place `p_end`.
    - **Activity Nodes**  
      - Each activity that passes filtering is added as an event node, annotated with:  
        - Normalized frequency  
        - Absolute frequency  
        - SPM quality score
      - Activities with no inputs, no outputs, and not marked as start/end are skipped.
    - **Input Places (joins)**  
      - For each activity, an intermediate place node is created for every INPUT set.  
      - Edges are drawn from each predecessor in the INPUT set to the place, and from the place to the activity.  
      - This models AND/OR joins.
      - If an input set contains the activity itself, a **dedicated self-loop place** is created.
    - **Output Places (splits)**  
      - For each activity, an intermediate place node is created for every OUTPUT set.  
      - Edges are drawn from the activity to the place, and from the place to each successor in the OUTPUT set.  
      - This models AND/OR splits.
      - If an output set contains the activity itself, a **dedicated self-loop place** is created.
    - **Overlap Handling**
      - To avoid redundancy, single-predecessor relations that are already covered by a larger OR/AND set are skipped.
  - **Consistency**  
    - Before rendering, causal relations `C` are rebuilt from the OUTPUT sets to ensure the graph reflects the most recent state of the model.
  - **Result**
    - Filtering thresholds (SPM and node frequency) ensure that only significant activities and relations appear in the final model.

---

## Dependency Initialization & Heuristic Creation

Before evolution starts, the algorithm computes dependency-based measures from the log and uses them to **probabilistically initialize individuals**.
- **Dependency measure**:

  $$
  D(a, b) =
  \begin{cases}
    \frac{L2L(a,b) + L2L(b,a)}{L2L(a,b) + L2L(b,a) + 1}, & \text{if } a \neq b \text{ and L2L(a,b) > 0} \\
    \frac{follows(a,b) - follows(b,a)}{follows(a,b) + follows(b,a) + 1}, & \text{if } a \neq b \text{ and L2L(a,b) = 0} \\
    \frac{L1L(a)}{L1L(a) + 1}, & \text{if } a = b
  \end{cases}
  $$

- **Start measure** for activity *t*:  

  $$
  S(t) =
  \begin{cases}
    \frac{L2L(start,t) + L2L(t,start)}{L2L(start,t) + L2L(t,start) + 1}, & \text{if L2L(start,t) > 0} \\
    \frac{follows(start,t) - follows(t,start)}{follows(start,t) + follows(t,start) + 1}, & \text{otherwise}
  \end{cases}
  $$

- **End measure** for activity *t*:  

  $$
  E(t) =
  \begin{cases}
    \frac{L2L(t,end) + L2L(end,t)}{L2L(t,end) + L2L(end,t) + 1}, & \text{if L2L(t,end) > 0} \\
    \frac{follows(t,end) - follows(end,t)}{follows(t,end) + follows(end,t) + 1}, & \text{otherwise}
  \end{cases}
  $$

These measures guide the **heuristic creation of individuals**:  
Activities with higher dependency/start/end likelihoods are more often connected, and rarity of causal relations is controlled by the power value.

---

## Filtering

The **SPM Filter (Search Process Model)** simplifies process models by removing low-quality nodes based on their frequency and connectivity. It scores each node using the spm metric, which balances complexity and common behavior. This abstraction helps generate clearer, more interpretable models which is especially useful for user-driven processes like search behavior.

The **SPM value** ranges from 0.0 to 1.0 and reflects the semantic quality of a node. It defines the threshold below which nodes are considered low-quality due to low frequency or high complexity, and may be removed for abstraction.

*For more information on the SPM metric, see: Marian Lux, Stefanie Rinderle-Ma, Andrei Preda: “Assessing the Quality of Search Process Models.”  
Available online: https://ucrisportal.univie.ac.at/en/publications/assessing-the-quality-of-search-process-models*

**Node frequency (normalized)**  
Measures how often each event (node) appears relative to the most frequent node in the (SPM-filtered) log.  
The result is a value between 0.0 and 1.0. Nodes with a normalized frequency below the threshold are removed.  

**Node frequency (absolute)**  
Counts the total number of occurrences of each event (node) in the log. Nodes with an absolute frequency below the threshold are removed.  

*Both thresholds are synchronized: adjusting one automatically updates the other, so the normalized and absolute values always stay consistent.*

**Power value**: odd integer controlling rarity of causal relations in dependency initialization.
  - `p=1` -> dense initialization  
  - `p=9` -> sparse initialization
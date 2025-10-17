import random
import threading
import uuid

import numpy as np

from graphs.visualization.genetic_graph import GeneticGraph
from logger import get_logger
from mining_algorithms.base_mining import BaseMining


class GeneticMining(BaseMining):
    """
    Genetic Mining algorithm implementation following Alves de Medeiros et al. (2005):
    Using Genetic Algorithms to Mine Process Models: Representation, Operators, and Results.

    This class applies a genetic algorithm to discover process models from event logs.
    Individuals are represented using a list of activities and INPUT, OUTPUT, causal sets,
    derived from dependency measures. Initialization is stochastic and controlled by an
    odd power parameter (power_value), which influences rarity of causal relations.
    The algorithm evolves a population using elitism, tournament selection, crossover, and mutation.
    Fitness evaluation is based on parsing semantics: the fraction of events
    successfully parsed and the fraction of traces properly completed.
    The best individual is translated into a graph for visualization.
    """

    _global_current_run_id = None
    _global_lock = threading.Lock()

    def __init__(self, log):
        """
        Initialize a GeneticMining instance.

        Parameters
        ----------
        log : dict[tuple[str, ...], int]
            A dictionary containing the traces and their frequencies in the log.
        """
        super().__init__(log)

        self.population_size = 500
        self.max_generations = 100
        self.crossover_rate = 1.0
        self.mutation_rate = 0.01
        self.elitism_rate = 0.01
        self.tournament_size = 5
        self.power_value = 1
        self.fitness_threshold = 1.0

        self.dependency_matrix = {}
        self.best_individual = None
        self.start_measures = {}
        self.end_measures = {}
        
        self.petri_net = None


        self.logger = get_logger("GeneticMining")

    def generate_graph(self, spm_threshold, node_freq_threshold_normalized, node_freq_threshold_absolute,
                       population_size, max_generations, crossover_rate, mutation_rate, elitism_rate, tournament_size,
                       power_value, fitness_threshold):
        """
        Generate a process model graph using the genetic miner.
        This method runs the full pipeline:
            Dependency matrix initialization -> Genetic evolution -> Fitness evaluation -> Termination -> graph construction.

        Parameters
        ----------
        spm_threshold : float
            Threshold for SPM filtering.
        node_freq_threshold_normalized : float
            Threshold for normalized node frequency filtering.
        node_freq_threshold_absolute : int
            Threshold for absolute node frequency filtering.
        population_size : int
            Number of individuals per generation.
        max_generations : int
            Maximum number of generations to evolve.
        crossover_rate : float
            Probability of crossover between parents.
        mutation_rate : float
            Probability of mutation per activity.
        elitism_rate : float
            Fraction of elite individuals copied unchanged.
        tournament_size : int
            Number of competitors in tournament selection.
        power_value : int
            Odd integer controlling rarity in initialization.
        fitness_threshold : float
            Minimum fitness value at which the heuristic individual creation stops early.
        """
        run_id = str(uuid.uuid4())

        with GeneticMining._global_lock:
            GeneticMining._global_current_run_id = run_id

        self.dependency_matrix = {}
        self.best_individual = None
        self.start_measures = {}
        self.end_measures = {}

        self.logger.debug(
            f"[generate_graph] spm_threshold={spm_threshold}, node_freq_threshold_normalized={node_freq_threshold_normalized}, "
            f"node_freq_threshold_absolute={node_freq_threshold_absolute}, population_size={population_size}, max_generations={max_generations}, "
            f"crossover_rate={crossover_rate}, mutation_rate={mutation_rate}, "
            f"elitism_rate={elitism_rate}, tournament_size={tournament_size},"
            f"power_value={power_value}"
        )

        self.spm_threshold = spm_threshold
        self.node_freq_threshold_normalized = node_freq_threshold_normalized
        self.node_freq_threshold_absolute = node_freq_threshold_absolute
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.power_value = power_value
        self.fitness_threshold = fitness_threshold

        self.recalculate_model_filters()

        if not self.filtered_events:
            self.graph = GeneticGraph()
            self.graph.add_start_node()
            self.graph.add_end_node()
            self.graph.create_edge("Start", "End")
            return

        self._initialize_dependency_matrix()

        self._run_genetic_miner(population_size, max_generations, crossover_rate, mutation_rate, elitism_rate,
                                tournament_size, run_id, power_value, fitness_threshold)

        # Only generate if it's still an active run
        with GeneticMining._global_lock:
            if GeneticMining._global_current_run_id != run_id:
                self.logger.debug("[generate_graph] Skipped graph generation (run cancelled)")
                return

        self._generate_graph_from_genetic()

    def _initialize_dependency_matrix(self):
        """
        Compute dependency, start, and end measures from the log traces.

        Dependency measures capture causal likelihood between activities, handling loops of length one and two.
        - L2L(a, b): number of length-two loops (a -> b -> a)
        - follows(a, b): number of times a is directly followed by b
        - L1L(a): number of self-loops (a -> a)

        Dependency measure D(a,b) is computed as:
        - Case 1 (length-two loop): (L2L(a,b) + L2L(b,a)) / (L2L(a,b) + L2L(b,a) +1)
        - Case 2 (no length-two loop, a != b): (follows(a,b) - follows(b,a)) / (follows(a,b) + follows(b,a) +1)
        - Case 3 (length-one loop (self-loop), a == b): L1L(a) / (L1L(a) +1)

        Start and end measures identify likely start and end activities.
        - Start measure: S(t) = D(start,t)
        - End measure:   E(t) = D(t,end)

        Returns
        -------
        None
            Updates self.dependency_matrix, self.start_measures, and self.end_measures.
        """
        all_activities = ['start'] + list(self.filtered_events) + ['end']
        activity_idx = {a: i for i, a in enumerate(all_activities)}
        n = len(all_activities)

        follows = np.zeros((n, n), dtype=np.int32)
        L1L = np.zeros(n, dtype=np.int32)
        L2L = np.zeros((n, n), dtype=np.int32)

        # Count over all traces
        for trace in self.node_frequency_filtered_log:
            # Add virtual start and end activities to each trace
            extended_trace = ['start'] + list(trace) + ['end']
            idx = np.fromiter((activity_idx[x] for x in extended_trace), count=len(extended_trace), dtype=np.int32)

            # follows and L1L
            if idx.size >= 2:
                a, b = idx[:-1], idx[1:]
                np.add.at(follows, (a, b), 1)
                np.add.at(L1L, a[a == b], 1)

            # L2L
            if idx.size >= 3:
                a, b, c = idx[:-2], idx[1:-1], idx[2:]
                mask = (a == c) & (a != b)
                np.add.at(L2L, (a[mask], b[mask]), 1)

        start, end = activity_idx['start'], activity_idx['end']

        # Initialize dependency matrix and start/end measures for all activity pairs
        for a in self.filtered_events:
            idx_a = activity_idx[a]
            for b in self.filtered_events:
                idx_b = activity_idx[b]
                if a != b:
                    l2l_ab = L2L[(idx_a, idx_b)]
                    l2l_ba = L2L[(idx_b, idx_a)]
                    if l2l_ab > 0:
                        # Case 1: Length-two loop exists
                        self.dependency_matrix[(a, b)] = (l2l_ab + l2l_ba) / (l2l_ab + l2l_ba + 1.0)
                    else:
                        # Case 2: No length-two loop
                        fab = follows[(idx_a, idx_b)]
                        fba = follows[(idx_b, idx_a)]
                        self.dependency_matrix[(a, b)] = (fab - fba) / (fab + fba + 1.0)
                else:
                    # Case 3: Length-one loop (a == b)
                    l1l = L1L[idx_a]
                    self.dependency_matrix[(a, b)] = l1l / (l1l + 1.0)

            # Calculate S(t) = D(start,t), E(t) = D(t,end)
            # S(t)
            l2l_sa = L2L[(start, idx_a)]
            l2l_as = L2L[(idx_a, start)]
            if l2l_sa > 0:
                self.start_measures[a] = (l2l_sa + l2l_as) / (l2l_sa + l2l_as + 1.0)
            else:
                f_sa = follows[(start, idx_a)]
                f_as = follows[(idx_a, start)]
                self.start_measures[a] = (f_sa - f_as) / (f_sa + f_as + 1.0)

            # E(t)
            l2l_ae = L2L[(idx_a, end)]
            l2l_ea = L2L[(end, idx_a)]
            if l2l_ae > 0:
                self.end_measures[a] = (l2l_ae + l2l_ea) / (l2l_ae + l2l_ea + 1.0)
            else:
                f_ae = follows[(idx_a, end)]
                f_ea = follows[(end, idx_a)]
                self.end_measures[a] = (f_ae - f_ea) / (f_ae + f_ea + 1.0)

    def _run_genetic_miner(self, population_size, max_generations, crossover_rate, mutation_rate, elitism_rate,
                           tournament_size, run_id, power_value, fitness_threshold):
        """
        Execute the genetic algorithm loop to evolve process models.

        Parameters
        ----------
        population_size : int
            Number of individuals per generation.
        max_generations : int
            Maximum number of generations to evolve.
        crossover_rate : float
            Probability of crossover between parents.
        mutation_rate : float
            Probability of mutation per activity.
        elitism_rate : float
            Fraction of elite individuals copied unchanged.
        tournament_size : int
            Number of competitors in tournament selection.
        run_id : str
            Unique identifier to cancel/track runs.
        power_value : int
            Odd integer controlling rarity in initialization.
        fitness_threshold : float
            Minimum fitness value at which the heuristic individual creation stops early.

        Returns
        -------
        None
            Updates `self.best_individual`.
        """
        best_so_far = -1.0
        generations = 0
        gen_limit = max_generations // 2
        stop_early = False

        self.logger.debug(f"[GeneticMiner] Starting run_id={run_id}")
        elitism_count = max(1, int(population_size * elitism_rate))

        init_population = [
            self._create_heuristic_individual(self.filtered_events, power_value) for _ in range(population_size)]

        for generation in range(max_generations):
            # Check if this run is still valid
            with GeneticMining._global_lock:
                if GeneticMining._global_current_run_id != run_id:
                    self.logger.debug(f"[GeneticMiner] Cancelled run_id={run_id} during generation {generation}")
                    stop_early = True
                    break

            if generation == 0:
                for ind in init_population:
                    if self._check_and_update_best_individual(ind, fitness_threshold, generation):
                        stop_early = True
                        break
                if stop_early:
                    break
                init_population.sort(key=lambda x: -x['fitness'])

            best_fitness = init_population[0]['fitness']

            if best_fitness > best_so_far:
                best_so_far = best_fitness
                generations = 0
            else:
                generations += 1

            self.logger.debug(
                f"Generation {generation}: fitness = {best_fitness:.4f}")

            if generations == gen_limit:
                self.logger.debug(f"[GeneticMiner] Stopped due to stagnation ({generations} == {gen_limit})")
                break

            next_gen = init_population[:elitism_count]

            while len(next_gen) < population_size and not stop_early:
                parent1 = self._tournament_selection(init_population, min(tournament_size, population_size))
                parent2 = self._tournament_selection(init_population, min(tournament_size, population_size))

                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = self._clone_individual(parent1), self._clone_individual(parent2)

                if mutation_rate > 0.0:
                    self._mutate(child1, mutation_rate)
                    self._mutate(child2, mutation_rate)

                # evaluate fitness
                if self._check_and_update_best_individual(child1, fitness_threshold, generation):
                    stop_early = True
                    break
                next_gen.append(child1)

                if len(next_gen) < population_size:
                    if self._check_and_update_best_individual(child2, fitness_threshold, generation):
                        stop_early = True
                        break
                    next_gen.append(child2)

            init_population = sorted(next_gen, key=lambda x: -x['fitness'])

            if stop_early:
                break

        if not self.best_individual and init_population:
            self.best_individual = init_population[0]

        self.logger.debug(f"[GeneticMiner] Completed run_id={run_id}")
        self.logger.debug(f"[FINAL CAUSAL MATRIX] C = {self.best_individual['C']}")
        self.logger.debug(f"[FINAL INDIVIDUAL] I = {self.best_individual['I']}")
        self.logger.debug(f"[FINAL INDIVIDUAL] O = {self.best_individual['O']}")
        self.logger.debug(f"[LOG TRACES] = {', '.join(str(trace) for trace in self.node_frequency_filtered_log)}")

    def _check_and_update_best_individual(self, ind, fitness_threshold, generation):
        """
        Evaluate the fitness of an individual and check against the threshold.

        - Update 'self.best_individual'.
        - Return True to signal that evolution should stop.

        Parameters
        ----------
        ind : dict
            Individual to evaluate.
        fitness_threshold : float
            Minimum fitness value at which termination is triggered.
        generation : int
            Current generation index.

        Returns
        -------
        bool
            True if stop condition is met, False otherwise.
        """
        ind['fitness'] = self._evaluate_fitness(ind)
        if ind['fitness'] >= fitness_threshold:
            self.best_individual = ind
            self.logger.debug(
                f"[GeneticMiner] Stopped early: fitness {ind['fitness']:.4f} ≥ {fitness_threshold} (Generation: {generation})")
            return True
        return False

    @staticmethod
    def _tournament_selection(population, tournament_size):
        """
        Select the fittest individual among a random tournament subset.

        Parameters
        ----------
        population : list[dict]
            Current population of individuals.
        tournament_size : int
            Number of individuals sampled for the tournament.

        Returns
        -------
        dict
            The selected fittest individual.
        """
        return max(random.sample(population, tournament_size), key=lambda x: x['fitness'])

    def _generate_graph_from_genetic(self):
        """
        Construct graph from the best discovered individual.

        This includes:
        - Adding start/end nodes
        - Creating places and edges for AND/OR splits/joins
        - Respecting filtering thresholds

        Returns
        -------
        None
            Updates `self.graph` with visualization.
        """
        self.logger.debug("[Graph] Generating graph from best individual...")
        self.graph = GeneticGraph()
        self.graph.add_start_node()
        self.graph.add_end_node()

        individual = self.best_individual
        if not individual:
            self.logger.debug("[Graph] Skipped: no valid individual to render.")
            return

        activities, C, I, O = individual['activities'], individual['C'], individual['I'], individual['O']

        self._rebuild_causal_relations(individual)

        node_stats_map = {stat['node']: stat for stat in self.get_node_statistics()}

        start_activities = [a for a in activities if a in self.start_nodes]
        end_activities = [a for a in activities if a in self.end_nodes]

        self.logger.debug(f"Start activities: {start_activities}")
        self.logger.debug(f"End activities: {end_activities}")

        for act in activities:
            if act not in self.filtered_events:
                continue

            # Skip isolated activities (no inputs, no outputs)
            if (not I.get(act) or all(len(s) == 0 for s in I[act])) and (
                    not O.get(act) or all(len(s) == 0 for s in O[act])) and not (
                    act in self.start_nodes or act in self.end_nodes):
                self.logger.debug(f"[Graph] Skipping isolated activity: {act}")
                continue

            stats = node_stats_map.get(act, {})
            abs_freq = self.filtered_appearance_freqs.get(act, 0)
            self.graph.add_event(
                str(act),
                spm=stats.get("spm", 0.0),
                normalized_frequency=stats.get("frequency", 0.0),
                absolute_frequency=abs_freq
            )

        # Place & Edge creation
        self.created_places = set()
        self.created_edges = set()

        # Pre-calc covered relations for OR/AND
        covered_relations = set()
        for act in activities:
            if act not in self.filtered_events:
                continue
            # Input
            for inp_set in I[act]:
                if len(inp_set) > 1:
                    for pred in inp_set:
                        covered_relations.add((pred, act))
            # Output
            for out_set in O[act]:
                if len(out_set) > 1:
                    for succ in out_set:
                        covered_relations.add((act, succ))

        for act in activities:
            if act not in self.filtered_events:
                continue

            # OUTPUT PLACES
            for out in O[act]:
                if not out:
                    continue

                # If the output set contains the activity itself, split it:
                if act in out:
                    self._ensure_self_loop_place(act)
                    rest = set(out) - {act}
                    if rest:
                        place_id = f"p_{act}_{'-'.join(sorted(rest))}"
                        if place_id not in self.created_places:
                            self.graph.add_place(place_id)
                            self._safe_create_edge(act, place_id, self.created_edges)
                            self.created_places.add(place_id)
                        for succ in rest:
                            self._safe_create_edge(place_id, succ, self.created_edges)
                    continue

                if len(out) == 1:
                    succ = next(iter(out))
                    if (act, succ) in covered_relations:
                        continue

                place_id = f"p_{act}_{'-'.join(sorted(out))}"
                if place_id not in self.created_places:
                    self.graph.add_place(place_id)
                    self._safe_create_edge(act, place_id, self.created_edges)
                    self.created_places.add(place_id)
                for succ in out:
                    self._safe_create_edge(place_id, succ, self.created_edges)

            # INPUT PLACES
            for inp in I[act]:
                if not inp:
                    continue

                # If the input set contains the activity itself, split it:
                if act in inp:
                    self._ensure_self_loop_place(act)
                    rest = set(inp) - {act}
                    if rest:
                        place_id = f"p_{'-'.join(sorted(rest))}_{act}"
                        if place_id not in self.created_places:
                            self.graph.add_place(place_id)
                            for pred in rest:
                                self._safe_create_edge(pred, place_id, self.created_edges)
                                self.logger.debug(f"[Graph] Edge: {pred} → {place_id}")
                            self.created_places.add(place_id)
                        self._safe_create_edge(place_id, act, self.created_edges)
                    continue

                if len(inp) == 1:
                    pred = next(iter(inp))
                    if (pred, act) in covered_relations:
                        continue

                place_id = f"p_{'-'.join(sorted(inp))}_{act}"
                if place_id not in self.created_places:
                    self.graph.add_place(place_id)
                    for pred in inp:
                        self._safe_create_edge(pred, place_id, self.created_edges)
                        self.logger.debug(f"[Graph] Edge: {pred} → {place_id}")
                    self.created_places.add(place_id)
                self._safe_create_edge(place_id, act, self.created_edges)

        place = f"p_start"
        self.graph.add_place(place)
        self._safe_create_edge("Start", place, self.created_edges)
        for act in start_activities:
            self._safe_create_edge(place, act, self.created_edges)
            self.logger.debug(f"[Graph] Connected Start → {act} via {place}")

        place = f"p_end"
        self.graph.add_place(place)
        for act in end_activities:
            self._safe_create_edge(act, place, self.created_edges)
            self.logger.debug(f"[Graph] Connected {act} → End via {place}")
        self._safe_create_edge(place, "End", self.created_edges)

        self.logger.debug("[Graph] Finished.")

    def _build_petri_net(self, individual):
        net = {
            'places': set(),
            'transitions': {},
            'arcs': set(),
        }

        




    def _ensure_self_loop_place(self, a: str):
        place_id = f"p_self_{a}"
        if place_id not in self.created_places:
            self.graph.add_place(place_id)
            self.created_places.add(place_id)
        self._safe_create_edge(a, place_id, self.created_edges)
        self._safe_create_edge(place_id, a, self.created_edges)
        self.logger.debug(f"[Graph] Self-loop place ensured for {a}: {place_id}")
        return place_id

    def _safe_create_edge(self, source, target, created_edges):
        if self.graph.contains_node(source) and self.graph.contains_node(target):
            edge = (source, target)
            if edge not in created_edges:
                self.graph.create_edge(source, target)
                created_edges.add(edge)
                self.logger.debug(f"[Graph] Edge: {source} → {target}")
        else:
            self.logger.debug(f"[Graph] Skipped edge {source} → {target} (missing node)")

    def _create_heuristic_individual(self, activities, power_value):
        """
        Create a new individual guided by dependency measures.

        Steps:
        1. Randomize causal matrix using dependency values.
        2. Apply start wipes.
        3. Apply end wipes.
        4 & 5. Build INPUT and OUTPUT sets and create partitions.
        6. Derive causal relation set C.

        Parameters
        ----------
        activities : list[str]
            List of all activities.
        power_value : int
            Odd power value controlling rarity of causal relations.

        Returns
        -------
        dict
            Individual with keys: 'activities', 'C', 'I', 'O'.
        """
        n = len(activities)
        activity_idx = {a: i for i, a in enumerate(activities)}

        # Build D matrix from dict
        D = np.zeros((n, n), dtype=float)
        for (a, b), val in self.dependency_matrix.items():
            if a in activity_idx and b in activity_idx:
                D[activity_idx[a], activity_idx[b]] = val

        # Step 1: probabilistic causal matrix
        causal_matrix = (np.random.rand(n, n) < (np.clip(D, 0.0, 1.0) ** power_value)).astype(int)

        # Step 2: apply S(t) wipe (columns)
        for a, idx_a in activity_idx.items():
            if np.random.rand() < (self.start_measures.get(a, 0.0) ** power_value):
                causal_matrix[:, idx_a] = 0

        # Step 3: apply E(t) wipe (rows)
        for a, idx_a in activity_idx.items():
            if np.random.rand() < (self.end_measures.get(a, 0.0) ** power_value):
                causal_matrix[idx_a, :] = 0

        # Step 4 & 5: build INPUT & OUTPUT sets
        I = {}
        O = {}
        for a, idx_a in activity_idx.items():
            preds = {activities[j] for j in np.where(causal_matrix[:, idx_a] == 1)[0]}
            I[a] = [s for s in self._create_partition(preds) if s]

            succs = {activities[j] for j in np.where(causal_matrix[idx_a, :] == 1)[0]}
            O[a] = [s for s in self._create_partition(succs) if s]

        # Step 6: derive C from O
        C = set()
        for a in activities:
            for out_set in O[a]:
                for b in out_set:
                    C.add((a, b))

        return {'activities': activities, 'C': C, 'I': I, 'O': O}

    @staticmethod
    def _create_partition(activities):
        """
        Randomly partition a set of activities into subsets.

        Parameters
        ----------
        activities : set[str]
            Activities to partition.

        Returns
        -------
        list[set[str]]
            Partition of activities into non-empty subsets.
        """
        if not activities:
            return [set()]

        activities = list(activities)
        random.shuffle(activities)

        k = random.randint(1, len(activities))
        partition = [set() for _ in range(k)]

        for i, elem in enumerate(activities):
            partition[i % k].add(elem)

        return partition

    def _evaluate_fitness(self, individual):
        """
        Evaluate fitness of an individual using continuous semantics.

        Fitness = 0.40 * (parsed activities / total activities) + 0.60 * (properly completed traces / total traces).

        Parameters
        ----------
        individual : dict
            Individual with activities, I/O sets, and C.

        Returns
        -------
        float
            Fitness score in [0.0, 1.0].
        """
        total_acts = max(1, sum(len(trace) for trace in self.node_frequency_filtered_log))
        total_traces = max(1, len(self.node_frequency_filtered_log))

        parsed_sum, completed_sum = 0, 0

        for trace in self.node_frequency_filtered_log:
            parsed, completed = self._parse_trace_token_game(individual, trace)
            parsed_sum += parsed
            completed_sum += 1 if completed else 0

        fitness = max(0.0, min(1.0, 0.40 * (parsed_sum / total_acts) + 0.60 * (completed_sum / total_traces)))

        return fitness

    def _parse_trace_token_game(self, individual, trace):
        """
        Simulate parsing of a single trace using token-game semantics.

        For each event in the trace:
        - INPUT semantics: an activity is enabled if all of its input sets are satisfied.
            * A single input set is satisfied if one of its providers currently holds a token
              (OR within a set, AND across sets).
            * Providers may be direct predecessors (AND-tokens), OR-bags, or start events.
        - When enabled, tokens are consumed from the chosen providers.
        - Tokens are produced to all OUTPUT sets.
            * A singleton output gives one token to the successor.
            * A multi-output set creates an OR-bag, which can be used once.
        - Self-loops are handled explicitly (tokens may remain on the same activity).
        - A trace is considered complete if all tokens are consumed at the end. -> NEEDS TO BE FIXED

        Parameters
        ----------
        individual : dict
            Individual with activities, I/O sets, and C.
            Must provide:
                - 'activities' : list[str]
                - 'I' : dict[str, list[set[str]]]  (input sets per activity)
                - 'O' : dict[str, list[set[str]]]  (output sets per activity)
        trace : list[str]
            Event sequence to parse.

        Returns
        -------
        tuple[int, bool]
            Number of parsed events and whether the trace completed properly.
        """
        I, O = individual['I'], individual['O']
        activities = individual['activities']

        # Precompute successor relations for faster lookup
        direct_successors = {act: set() for act in activities}  # singleton outputs
        or_output_groups = {act: [] for act in activities}  # multi-output OR groups
        for act in activities:
            for output_set in (O.get(act, []) or []):
                if not output_set:
                    continue
                if len(output_set) == 1:
                    direct_successors[act].add(next(iter(output_set)))
                else:
                    or_output_groups[act].append(tuple(sorted(output_set)))

        # Token bookkeeping
        tokens = {act: 0 for act in activities}
        available_or_bags = set()
        parsed_count = 0

        for event in trace:
            input_sets = I.get(event, []) or []
            output_sets = O.get(event, []) or []

            # Skip floating outputs (no real effect, unless it is an end node)
            has_real_outputs = any(out for out in output_sets if out)
            if (not output_sets or not has_real_outputs) and (event not in self.end_nodes):
                continue

            # Skip floating inputs (isolated event, unless start node)
            has_inputs = bool(input_sets)
            all_inputs_empty = has_inputs and all(len(s) == 0 for s in input_sets)
            if all_inputs_empty and (event not in self.start_nodes):
                continue

            # Track if event can fire
            can_execute = True
            chosen_providers = []
            start_self_available = (event in self.start_nodes)

            # Check each input set (AND across sets)
            if not input_sets:
                if event not in self.start_nodes:
                    can_execute = False
            else:
                for input_set in input_sets:
                    if not input_set:
                        if event in self.start_nodes:
                            chosen_providers.append(('none', None))
                            continue
                        else:
                            can_execute = False
                            break

                    provider = None

                    # Direct predecessor
                    for predecessor in input_set:
                        if (event in direct_successors.get(predecessor, set())
                                and tokens.get(predecessor, 0) > 0):
                            provider = ('token', predecessor)
                            break

                    # OR-bag
                    if provider is None:
                        for predecessor in input_set:
                            for group in or_output_groups.get(predecessor, []):
                                if event in group and (predecessor, group) in available_or_bags:
                                    provider = ('or', (predecessor, group))
                                    break
                            if provider:
                                break

                    # Start-self
                    if provider is None and (event in input_set) and start_self_available:
                        provider = ('start', event)
                        start_self_available = False

                    if provider is None:
                        can_execute = False
                        break
                    chosen_providers.append(provider)

            # Consume tokens if firing
            for relation, provider_data in chosen_providers:
                if relation == 'token':
                    predecessor = provider_data
                    tokens[predecessor] -= 1
                elif relation == 'or':
                    predecessor, group = provider_data
                    if (predecessor, group) in available_or_bags:
                        available_or_bags.remove((predecessor, group))

            if can_execute:
                parsed_count += 1

            # Produce outputs
            for output_set in output_sets:
                if not output_set:
                    continue
                if len(output_set) == 1:
                    tokens[event] += 1
                else:
                    group = tuple(sorted(output_set))
                    available_or_bags.add((event, group))

        # Full trace completion check
        is_completed = (
            parsed_count == len(trace)  
            and all(v == 0 for v in tokens.values())  
            and not available_or_bags  
        )

        # Check for L1L
        if any(
            trace[i] == trace[i - 1] and
            any(trace[i] in out for out in individual["O"].get(trace[i], []))
            for i in range(1, len(trace))
        ):
            is_completed = False

        return parsed_count, is_completed

    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents at a random activity.

        INPUT and OUTPUT partitions of the crossover activity are recombined
        using random swap points. Overlaps are resolved and consistency is restored.

        Parameters
        ----------
        parent1 : dict
            Parent individual 1.
        parent2 : dict
            Parent individual 2.

        Returns
        -------
        tuple[dict, dict]
            Two offspring individuals.
        """
        if parent1 == parent2:
            return self._clone_individual(parent1), self._clone_individual(parent2)

        offspring1 = self._clone_individual(parent1)
        offspring2 = self._clone_individual(parent2)

        # Select crossover activity
        t = random.choice(parent1['activities'])

        # Recombine INPUT(t)
        set1 = list(offspring1['I'][t])
        set2 = list(offspring2['I'][t])
        if set1 and set2:
            sp1 = random.randint(0, len(set1) - 1)
            sp2 = random.randint(0, len(set2) - 1)
            # swap suffixes
            new_in1 = set1[:sp1] + set2[sp2:]
            new_in2 = set2[:sp2] + set1[sp1:]
            # resolve overlaps
            offspring1['I'][t] = self._resolve_overlaps(new_in1)
            offspring2['I'][t] = self._resolve_overlaps(new_in2)

        # Recombine OUTPUT(t)
        out1 = list(offspring1['O'][t])
        out2 = list(offspring2['O'][t])
        if out1 and out2:
            sp1 = random.randint(0, len(out1) - 1)
            sp2 = random.randint(0, len(out2) - 1)
            new_out1 = out1[:sp1] + out2[sp2:]
            new_out2 = out2[:sp2] + out1[sp1:]
            offspring1['O'][t] = self._resolve_overlaps(new_out1)
            offspring2['O'][t] = self._resolve_overlaps(new_out2)

        # Ensure consistency
        self._update_related_activities(offspring1, t)
        self._update_related_activities(offspring2, t)

        return offspring1, offspring2

    @staticmethod
    def _clone_individual(ind):
        """
        Create a copy of an individual.

        Copies activities, causal set C, and input/output sets (I, O).
        Ensures that mutable structures (sets, lists) are duplicated to avoid side effects when modifying the clone.

        Parameters
        ----------
        ind : dict
            Individual represented with:
            - 'activities': list[str]
            - 'C': set[tuple[str, str]]
            - 'I': dict[str, list[set[str]]]
            - 'O': dict[str, list[set[str]]]
            - 'fitness': float

        Returns
        -------
        dict
             Copy of the individual with the same structure and values.
        """
        return {
            'activities': list(ind['activities']),
            'C': set(ind['C']),
            'I': {a: [set(input_sets) for input_sets in (ind['I'].get(a) or [set()])] for a in ind['activities']},
            'O': {a: [set(output_sets) for output_sets in (ind['O'].get(a) or [set()])] for a in ind['activities']},
            'fitness': ind.get('fitness', 0.0),
        }

    @staticmethod
    def _resolve_overlaps(subsets):
        """
        Resolve overlaps between subsets by merging or splitting.

        Ensures that the resulting partition of activities does not contain conflicting overlaps.
        When two subsets overlap, they are either merged into one or the overlap is removed, chosen at random.

        Parameters
        ----------
        subsets : list[set[str]]
            List of subsets representing input or output partitions.

        Returns
        -------
        list[set[str]]
            Updated list of disjoint subsets.
        """
        resolved = [set(s) for s in subsets if s]

        changed = True
        while changed:
            changed = False
            for i in range(len(resolved)):
                for j in range(i + 1, len(resolved)):
                    if not resolved[i].isdisjoint(resolved[j]):  # overlap found
                        if random.random() < 0.5:
                            resolved[i] |= resolved[j]  # merge
                            resolved.pop(j)
                        else:
                            resolved[j] -= resolved[i]  # remove
                            if not resolved[j]:
                                resolved.pop(j)
                        changed = True
                        break
                if changed:
                    break

        return resolved

    def _mutate(self, individual, mutation_rate):
        """
        Apply mutation to an individual`s I/O sets.

        For each activity, with probability 'mutation_rate':
        - Merge or split one of its input sets.
        - Merge or split one of its output sets.

        Parameters
        ----------
        individual : dict
            Individual to mutate.
        mutation_rate : float
            Probability of mutating each activity.

        Returns
        -------
        None
            Individual is modified in place.
        """
        for act in individual['activities']:
            if random.random() < mutation_rate:
                # Mutate INPUT sets
                if individual['I'][act]:
                    individual['I'][act] = self._mutate_sets(individual['I'][act])
                # Mutate OUTPUT sets
                if individual['O'][act]:
                    individual['O'][act] = self._mutate_sets(individual['O'][act])

        self._update_related_activities(individual)

    @staticmethod
    def _mutate_sets(sets_list):
        """
        Randomly merge or split subsets in a given partition.

        Parameters
        ----------
        sets_list : list[set[str]]
            Current subsets of activities.

        Returns
        -------
        list[set[str]]
            Updated list of subsets after mutation.
        """
        sets_list = [set(s) for s in sets_list if s]
        if len(sets_list) < 1:
            return sets_list

        if random.random() < 0.5 and len(sets_list) > 1:
            # MERGE
            a, b = random.sample(sets_list, 2)
            merged = a | b
            new_sets = [s for s in sets_list if s not in (a, b)]
            new_sets.append(merged)
            return new_sets
        else:
            # SPLIT
            s = random.choice(sets_list)
            if len(s) > 1:
                k = random.randint(1, len(s) - 1)
                set1 = set(random.sample(list(s), k))
                set2 = s - set1
                new_sets = [x for x in sets_list if x != s]
                new_sets.extend([set1, set2])
                return new_sets

        return sets_list

    def _update_related_activities(self, individual, t=None):
        """
        Update I/O sets to keep them consistent.

        Steps when t is given:
        1. Collect all predecessors of t from I(t) and successors of t from O(t).
        2. For every a != t:
           - If a is in I(t) but O(a) does not contain t -> add t to O(a).
           - If a is not in I(t) but O(a) contains t -> remove t from O(a).
           - Remove duplicate appearances of t inside O(a).
        3. For every b != t:
           - If b is in O(t) but I(b) does not contain t -> add t to I(b).
           - If b is not in O(t) but I(b) contains t -> remove t from I(b).
           - Remove duplicate appearances of t inside I(b).
        4. Remove any empty subsets created during the cleanup.
        5. Rebuild C from the updated O sets.

        Steps when t is None:
        1. For each relation a->b in O:
           - Ensure a is present in I(b).
        2. For each relation a->b in I:
           - Ensure b is present in O(a).
        3. Remove any relations that are inconsistent (present only on one side).
        4. Remove any empty subsets created during the cleanup.
        5. Rebuild C from the updated O sets.

        Parameters
        ----------
        individual : dict
            Individual with activities, I/O sets, and C.
        t : str or None
            Specific activity to check after crossover/mutation. If None, check all.

        Returns
        -------
        None
            Individual is modified in place.
        """
        I, O, C = individual['I'], individual['O'], individual['C']
        acts = individual['activities']

        def prune(lst):
            if lst:
                lst[:] = [s for s in lst if s]

        def add_to_random_subset(partitions, key, elem):
            lst = partitions.setdefault(key, [])
            for s in lst:
                if elem in s:
                    return
            nonempty = [s for s in lst if s]
            if nonempty:
                random.choice(nonempty).add(elem)
            else:
                lst.append({elem})

        if t is not None:
            preds = set().union(*(s for s in I.get(t, []) if s)) if I.get(t) else set()
            succs = set().union(*(s for s in O.get(t, []) if s)) if O.get(t) else set()

            # COLUMN repair: ensure all O(a) match I(t) (predecessors of t)
            for a in acts:
                if a == t:
                    continue
                want = a in preds
                has = any(t in s for s in O.get(a, []))
                if want and not has:
                    add_to_random_subset(O, a, t)
                elif not want and has:
                    for s in O[a]:
                        s.discard(t)
                    prune(O[a])
                # de-duplicate t across subsets
                seen = False
                for s in O.get(a, []):
                    if t in s:
                        if seen:
                            s.discard(t)
                        seen = True
                prune(O.get(a, []))

            # ROW repair: ensure all I(b) match O(t) (successors of t)
            for b in acts:
                if b == t:
                    continue
                want = b in succs
                has = any(t in s for s in I.get(b, []))
                if want and not has:
                    add_to_random_subset(I, b, t)
                elif not want and has:
                    for s in I[b]:
                        s.discard(t)
                    prune(I[b])
                # de-duplicate t across subsets
                seen = False
                for s in I.get(b, []):
                    if t in s:
                        if seen:
                            s.discard(t)
                        seen = True
                prune(I.get(b, []))

        else:
            # forward -> reverse
            for a, outs in O.items():
                for out in outs:
                    for b in list(out):
                        if all(a not in s for s in I.get(b, [])):
                            add_to_random_subset(I, b, a)
            # reverse -> forward
            for b, ins in I.items():
                for inn in ins:
                    for a in list(inn):
                        if all(b not in s for s in O.get(a, [])):
                            add_to_random_subset(O, a, b)
            # remove inconsistencies
            for a, outs in O.items():
                for out in outs:
                    out.difference_update({b for b in out if all(a not in s for s in I.get(b, []))})
                prune(outs)
            for b, ins in I.items():
                for inn in ins:
                    inn.difference_update({a for a in inn if all(b not in s for s in O.get(a, []))})
                prune(ins)

        # rebuild C from O
        self._rebuild_causal_relations(individual)

    @staticmethod
    def _rebuild_causal_relations(individual):
        """
        Derive causal relation set C from the OUTPUT sets.

        Parameters
        ----------
        individual : dict
            Process model with I/O sets.

        Returns
        -------
        None
            Updates individual['C'].
        """
        C = set()
        for a in individual['activities']:
            for out_set in individual['O'][a]:
                for b in out_set:
                        C.add((a, b))
        individual['C'] = C

    def get_population_size(self):
        return self.population_size

    def get_max_generations(self):
        return self.max_generations

    def get_crossover_rate(self):
        return self.crossover_rate

    def get_mutation_rate(self):
        return self.mutation_rate

    def get_elitism_rate(self):
        return self.elitism_rate

    def get_tournament_size(self):
        return self.tournament_size

    def get_power_value(self):
        return self.power_value
import random
import threading
import uuid

import streamlit as st

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
        self.dependency_matrix_fallback = {}
        self.dependency_matrix = {}
        self.start_measures = {}
        self.end_measures = {}

        self.best_individual_fallback = None
        self.best_individual = None

        self.logger = get_logger("GeneticMining")

    def generate_graph(self, spm_threshold, node_freq_threshold, population_size, max_generations, crossover_rate,
                       mutation_rate, elitism_rate, tournament_size, power_value):
        """
        Generate a process model graph using the genetic miner.
        This method runs the full pipeline:
            Dependency matrix initialization -> Genetic evolution -> Fitness evaluation -> Termination -> Filtering -> graph construction.

        Parameters
        ----------
        spm_threshold : float
            Threshold for SPM filtering.
        node_freq_threshold : float
            Threshold for normalized node frequency filtering.
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
        """
        run_id = str(uuid.uuid4())

        with GeneticMining._global_lock:
            GeneticMining._global_current_run_id = run_id

        if st.session_state.get("cancel_run", False):
            self.dependency_matrix = self.dependency_matrix_fallback
            self.best_individual = self.best_individual_fallback
            return

        if self.spm_threshold == spm_threshold and self.node_freq_threshold == node_freq_threshold:
            self.dependency_matrix_fallback = self.dependency_matrix
            self.dependency_matrix = {}
            self.best_individual_fallback = self.best_individual
            self.best_individual = None
            self.start_measures = {}
            self.end_measures = {}

        self.logger.debug(
            f"[generate_graph] spm_threshold={spm_threshold}, node_freq_threshold={node_freq_threshold}, "
            f"population_size={population_size}, max_generations={max_generations}, "
            f"crossover_rate={crossover_rate}, mutation_rate={mutation_rate}, "
            f"elitism_rate={elitism_rate}, tournament_size={tournament_size},"
            f"power_value={power_value}"
        )

        self.spm_threshold = spm_threshold
        self.node_freq_threshold = node_freq_threshold
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.power_value = power_value

        self.recalculate_model_filters()

        if not self.filtered_events:
            self.graph = GeneticGraph()
            self.graph.add_start_node()
            self.graph.add_end_node()
            self.graph.create_edge("Start", "End")
            return

        if not self.dependency_matrix:
            self._initialize_dependency_matrix()

        if not self.best_individual:
            self._run_genetic_miner(population_size, max_generations,
                                    crossover_rate, mutation_rate,
                                    elitism_rate, tournament_size, run_id, power_value)

        # Only generate if it's still an active run
        with GeneticMining._global_lock:
            if GeneticMining._global_current_run_id != run_id:
                self.logger.debug("[generate_graph] Skipped graph generation (run cancelled)")
                return

        self._generate_graph_from_genetic()

    def _initialize_dependency_matrix(self):
        """
        Compute dependency measures, start measures, and end measures from the log traces.
        - Dependency measures capture causal likelihood between activities, handling loops of length one and two.
        - Start and end measures identify likely start and end activities.

        Returns
        -------
        None
            Updates self.dependency_matrix, self.start_measures, and self.end_measures.
        """
        all_activities = ['start'] + list(self.events) + ['end']

        follows = {(a, b): 0 for a in all_activities for b in all_activities}
        L1L = {a: 0 for a in all_activities}
        L2L = {(a, b): 0 for a in all_activities for b in all_activities}

        for trace in self.log:
            # Add virtual start and end activities to each trace
            extended_trace = ['start'] + list(trace) + ['end']

            # Count follows and L1L
            for i in range(len(extended_trace) - 1):
                a, b = extended_trace[i], extended_trace[i + 1]
                follows[(a, b)] += 1
                if a == b:
                    L1L[a] += 1

            # Count L2L
            for i in range(len(extended_trace) - 2):
                a, b, c = extended_trace[i], extended_trace[i + 1], extended_trace[i + 2]
                if a == c and a != b:  # length-two-loop pattern "aba"
                    L2L[(a, b)] += 1

        # Initialize dependency matrix and start/end measures for all activity pairs
        for a in self.events:
            for b in self.events:
                if a != b:
                    l2l_ab = L2L[(a, b)]
                    l2l_ba = L2L[(b, a)]
                    if l2l_ab > 0:
                        # Case 1: Length-two loop exists
                        self.dependency_matrix[(a, b)] = (l2l_ab + l2l_ba) / (l2l_ab + l2l_ba + 1.0)
                    else:
                        # Case 2: No length-two loop
                        fab = follows[(a, b)]
                        fba = follows[(b, a)]
                        self.dependency_matrix[(a, b)] = (fab - fba) / (fab + fba + 1.0)
                else:
                    # Case 3: Length-one loop (a == b)
                    l1l = L1L[a]
                    self.dependency_matrix[(a, b)] = l1l / (l1l + 1.0)

            # Calculate S(t) = D(start,t), E(t) = D(t,end)
            # S(t)
            l2l_sa = L2L[('start', a)]
            l2l_as = L2L[(a, 'start')]
            if l2l_sa > 0:
                self.start_measures[a] = (l2l_sa + l2l_as) / (l2l_sa + l2l_as + 1.0)
            else:
                f_sa = follows[('start', a)]
                f_as = follows[(a, 'start')]
                self.start_measures[a] = (f_sa - f_as) / (f_sa + f_as + 1.0)

            # E(t)
            l2l_ae = L2L[(a, 'end')]
            l2l_ea = L2L[('end', a)]
            if l2l_ae > 0:
                self.end_measures[a] = (l2l_ae + l2l_ea) / (l2l_ae + l2l_ea + 1.0)
            else:
                f_ae = follows[(a, 'end')]
                f_ea = follows[('end', a)]
                self.end_measures[a] = (f_ae - f_ea) / (f_ae + f_ea + 1.0)

    def _run_genetic_miner(self, population_size, max_generations, crossover_rate, mutation_rate, elitism_rate,
                           tournament_size, run_id, power_value):
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

        Returns
        -------
        None
            Updates `self.best_individual`.
        """
        best_so_far = -1.0
        generations = 0
        gen_limit = max_generations // 2

        self.logger.debug(f"[GeneticMiner] Starting run_id={run_id}")
        elitism_count = max(1, int(population_size * elitism_rate))

        init_population = [
            self._create_heuristic_individual(self.events, power_value) for _ in range(population_size)]
        fitness_history = []

        for generation in range(max_generations):
            # Check if this run is still valid
            with GeneticMining._global_lock:
                if GeneticMining._global_current_run_id != run_id:
                    self.logger.debug(f"[GeneticMiner] Cancelled run_id={run_id} during generation {generation}")
                    return

            for ind in init_population:
                ind['fitness'] = self._evaluate_fitness(ind)
            init_population.sort(key=lambda x: -x['fitness'])

            best_fitness = init_population[0]['fitness']
            fitness_history.append(best_fitness)

            if best_fitness > best_so_far:
                best_so_far = best_fitness
                generations = 0
            else:
                generations += 1

            self.logger.debug(
                f"Generation {generation}: fitness = {best_fitness:.4f}")

            if best_fitness == 1.0:
                self.logger.debug(f"[GeneticMiner] Optimal fitness reached in run_id={run_id}")
                break

            if generations == gen_limit:
                self.logger.debug(f"[GeneticMiner] Stopped due to stagnation ({generations} == {gen_limit})")
                break

            next_gen = init_population[:elitism_count]

            while len(next_gen) < population_size:
                parent1 = self._tournament_selection(init_population, min(tournament_size, population_size))
                parent2 = self._tournament_selection(init_population, min(tournament_size, population_size))

                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = self._clone_individual(parent1), self._clone_individual(parent2)

                if mutation_rate > 0.0:
                    self._mutate(child1, mutation_rate)
                    self._mutate(child2, mutation_rate)

                next_gen.append(child1)
                if len(next_gen) < population_size:
                    next_gen.append(child2)

            init_population = next_gen

        self.best_individual = init_population[0]

        self.logger.debug(f"[GeneticMiner] Completed run_id={run_id}")
        self.logger.debug(f"[FINAL CAUSAL MATRIX] C = {self.best_individual['C']}")
        self.logger.debug(f"[FINAL INDIVIDUAL] I = {self.best_individual['I']}")
        self.logger.debug(f"[FINAL INDIVIDUAL] O = {self.best_individual['O']}")
        self.logger.debug(f"[LOG TRACES] = {', '.join(str(trace) for trace in self.node_frequency_filtered_log)}")

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

        individual = self._get_filtered_individual()
        if not individual:
            self.logger.debug("[Graph] Skipped: no valid individual to render.")
            return

        activities, C, I, O = individual['activities'], individual['C'], individual['I'], individual['O']

        self._rebuild_causal_relations(individual)

        node_stats_map = {stat['node']: stat for stat in self.get_node_statistics()}

        start_activities = []
        end_activities = []

        for act in activities:
            if act in self.filtered_events:
                # START activities
                if self._is_start_activity(act, I):
                    start_activities.append(act)

                # END activities
                if self._is_end_activity(act, O):
                    end_activities.append(act)

        self.logger.debug(f"Start activities: {start_activities}")
        self.logger.debug(f"End activities: {end_activities}")

        for act in activities:
            if act not in self.filtered_events:
                continue

            stats = node_stats_map.get(act, {})
            abs_freq = self.filtered_appearance_freqs.get(act, 0)
            self.graph.add_event(
                str(act),
                spm=stats.get("spm", 0.0),
                normalized_frequency=stats.get("frequency", 0.0),
                absolute_frequency=abs_freq
            )

        created_places = set()
        created_edges = set()

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

        # Place & Edge creation
        for act in activities:
            if act not in self.filtered_events:
                continue

            # INPUT PLACES
            for inp_set in I[act]:
                if not inp_set:
                    continue

                # Skip if already covered
                if len(inp_set) == 1:
                    pred = next(iter(inp_set))
                    if (pred, act) in covered_relations:
                        continue

                place_id = f"p_{'-'.join(sorted(inp_set))}_{act}"
                if place_id not in created_places:
                    self.graph.add_place(place_id)
                    for pred in inp_set:
                        edge = (pred, place_id)
                        if edge not in created_edges:
                            self.graph.create_edge(pred, place_id)
                            created_edges.add(edge)
                            self.logger.debug(f"[Graph] Edge: {pred} → {place_id}")
                    created_places.add(place_id)

                edge = (place_id, act)
                if edge not in created_edges:
                    self.graph.create_edge(place_id, act)
                    created_edges.add(edge)
                    self.logger.debug(f"[Graph] Edge: {place_id} → {act}")

            # OUTPUT PLACES
            for out_set in O[act]:
                if not out_set:
                    continue

                # Skip if already covered
                if len(out_set) == 1:
                    succ = next(iter(out_set))
                    if (act, succ) in covered_relations:
                        continue

                place_id = f"p_{act}_{'-'.join(sorted(out_set))}"
                if place_id not in created_places:
                    self.graph.add_place(place_id)
                    edge = (act, place_id)
                    if edge not in created_edges:
                        self.graph.create_edge(act, place_id)
                        created_edges.add(edge)
                        self.logger.debug(f"[Graph] Edge: {act} → {place_id}")
                    created_places.add(place_id)

                for succ in out_set:
                    edge = (place_id, succ)
                    if edge not in created_edges:
                        self.graph.create_edge(place_id, succ)
                        created_edges.add(edge)
                        self.logger.debug(f"[Graph] Edge: {place_id} → {succ}")

        place = f"p_start"
        self.graph.add_place(place)
        self.graph.create_edge("Start", place)
        for act in start_activities:
            self.graph.create_edge(place, act)
            self.logger.debug(f"[Graph] Connected Start → {act} via {place}")

        place = f"p_end"
        self.graph.add_place(place)
        for act in end_activities:
            self.graph.create_edge(act, place)
            self.logger.debug(f"[Graph] Connected {act} → End via {place}")
        self.graph.create_edge(place, "End")

        self.logger.debug("[Graph] Finished.")

    def _create_heuristic_individual(self, activities, power_value):
        """
        Create a new individual guided by dependency measures.

        Initialization procedure:
        1. Randomize causal matrix using dependency values.
        2. Apply start wipes.
        3. Apply end wipes.
        4 & 5. Build INPUT and OUTPUT and generate partitions.
        6. Derive causal relation set C.

        Parameters
        ----------
        activities : list[str]
            List of all activity labels.
        power_value : int
            Odd power value controlling rarity of causal relations.

        Returns
        -------
        dict
            Individual with keys: 'activities', 'C', 'I', 'O'.
        """
        causal_matrix = {}

        # Step 1: probabilistic causal matrix
        for a in activities:
            for b in activities:
                D_a_b = self.dependency_matrix.get((a, b), 0.0)
                r = random.random()
                if r < (D_a_b ** power_value):
                    causal_matrix[(a, b)] = 1
                else:
                    causal_matrix[(a, b)] = 0

        # Step 2: apply S(t) wipe (columns)
        for a in activities:
            r = random.random()
            if r < (self.start_measures.get(a, 0.0) ** power_value):
                for b in activities:
                    causal_matrix[(b, a)] = 0

        # Step 3: apply E(t) wipe (rows)
        for a in activities:
            r = random.random()
            if r < (self.end_measures.get(a, 0.0) ** power_value):
                for b in activities:
                    causal_matrix[(a, b)] = 0

        # Step 4 & 5: build INPUT & OUTPUT sets
        I = {}
        O = {}
        for a in activities:
            preds = {b for b in activities if causal_matrix.get((b, a), 0) == 1}
            I[a] = [s for s in self._create_partition(preds) if s]

            succs = {b for b in activities if causal_matrix.get((a, b), 0) == 1}
            O[a] = [s for s in self._create_partition(succs) if s]

        # Step 6: derive C from O
        C = set()
        for a in activities:
            for out_set in O[a]:
                for b in out_set:
                    if a != b:
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

        return max(0.0, min(1.0, 0.40 * (parsed_sum / total_acts) + 0.60 * (completed_sum / total_traces)))

    @staticmethod
    def _parse_trace_token_game(individual, trace):
        """
        Parse a single trace using a token-game simulation.

        Each event is checked against INPUT sets, consumes/produces tokens
        via OUTPUT sets, and OR-splits/joins are tracked explicitly.
        Completion is defined as no tokens left at the end.

        Note
        ----
        In Alves de Medeiros et al. (2005, Sec. 2.2), an activity is enabled if
        at least one element of each INPUT subset holds a token (AND-of-OR semantics).
        Here, the check is inverted: execution is allowed as long as no conflicting
        tokens are present (self-loops never block). This differs from the formal
        description but results in equivalent behavior in practice, while simplifying
        the handling of L1L/L2L and OR/AND cases.

        Parameters
        ----------
        individual : dict
            Individual with activities, I/O sets, and C.
        trace : list[str]
            Event sequence to parse.

        Returns
        -------
        tuple[int, bool]
            Number of parsed events and whether the trace completed properly.
        """
        I, O = individual['I'], individual['O']
        marking = {a: 0 for a in individual['activities']}
        parsed_count = 0
        saved_or_cases = []  # track active (X)OR-branches

        for event in trace:
            input_sets = I.get(event, [])

            if (not input_sets or set() in input_sets) and marking[event] == 0:
                marking[event] = 1

            # Check if event is enabled and input sets are satisfied
            can_execute = False
            if marking.get(event, 0) > 0:  # must be enabled by predecessor
                ok = True
                for subset in input_sets:
                    if subset:
                        # Special-case: self-loop must not block
                        effective_subset = subset - {event}
                        if any(marking.get(x, 0) > 0 for x in effective_subset):
                            ok = False
                            break
                if ok:
                    can_execute = True

            if not can_execute:
                continue

            # Consume the event
            marking[event] = 0

            # Produce tokens
            for out_set in O.get(event, []):
                if not out_set:
                    continue
                if len(out_set) > 1:
                    # OR-split: save the OR-case for later resolution
                    saved_or_cases.append(out_set)
                for succ in out_set:
                    marking[succ] = 1

            # OR-join resolution
            for or_set in saved_or_cases[:]:
                if event in or_set:
                    for s in or_set:
                        marking[s] = 0  # disable all others in the OR-branch
                    saved_or_cases.remove(or_set)

            parsed_count += 1

        is_completed = (sum(marking.values()) == 0)
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
        self._update_related_activities(offspring1)
        self._update_related_activities(offspring2)

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
            Updated list of subsets with overlaps resolved.
        """
        resolved = []

        for s in subsets:
            s = set(s)
            if not s:
                continue

            while True:
                overlap_found = False
                for r in resolved:
                    if s & r:  # overlap found
                        overlap_found = True
                        if random.random() < 0.5:
                            r |= s  # merge
                            s = set()  # s is consumed
                            break
                        else:
                            s -= r  # remove overlaps from s
                if not overlap_found:
                    break
            if s:
                resolved.append(s)

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

    @staticmethod
    def _update_related_activities(individual):
        """
        Ensure INPUT and OUTPUT sets remain consistent.

        Enforces:
        - b in O(a) <=> a in I(b)
        - Removes outdated links

        Parameters
        ----------
        individual : dict
            Individual with activities, I/O sets, and C.

        Returns
        -------
        dict
            Updated individual with consistent I/O sets.
        """
        # forward -> reverse consistency
        for a, out in list(individual['O'].items()):
            for out_set in list(out):
                for b in list(out_set):
                    inp_b = individual['I'].setdefault(b, [set()])
                    if not any(a in subset for subset in inp_b):
                        empty = next((s for s in inp_b if not s), None)
                        if empty is not None:
                            empty.add(a)
                        else:
                            inp_b.append({a})

        # reverse -> forward consistency
        for b, inp in list(individual['I'].items()):
            for inp_set in list(inp):
                for a in list(inp_set):
                    out_a = individual['O'].setdefault(a, [set()])
                    if not any(b in subset for subset in out_a):
                        empty = next((s for s in out_a if not s), None)
                        if empty is not None:
                            empty.add(b)
                        else:
                            out_a.append({b})

        # remove inconsistencies
        #  - If a not in I(b) then remove b from O(a)
        #  - If b not in O(a) then remove a from I(b)
        for a, out in list(individual['O'].items()):
            for out_set in list(out):
                for b in list(out_set):
                    inp_b = individual['I'].get(b, [])
                    if not any(a in subset for subset in inp_b):
                        out_set.discard(b)

        for b, inp in list(individual['I'].items()):
            for inp_set in list(inp):
                for a in list(inp_set):
                    out_a = individual['O'].get(a, [])
                    if not any(b in subset for subset in out_a):
                        inp_set.discard(a)

        return individual

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
                    if a != b:
                        C.add((a, b))
        individual['C'] = C

    @staticmethod
    def _is_start_activity(act, I):
        """
        Check whether an activity is a start activity.

        Returns
        -------
        bool
            True if start activity, False otherwise.
        """
        inp = I.get(act, None)
        if not inp:
            return False
        return any(len(subset) == 0 for subset in inp)

    @staticmethod
    def _is_end_activity(act, O):
        """

        Check whether an activity is an end activity.

        Returns
        -------
        bool
            True if end activity, False otherwise.
        """
        out = O.get(act, None)
        if not out:
            return False
        return any(len(subset) == 0 for subset in out)

    def _get_filtered_individual(self):
        """
        Return best individual restricted to currently filtered events.

        Filters out any I/O subsets containing removed events and
        inserts empty sets if needed to preserve start/end semantics.

        Returns
        -------
        dict | None
            Filtered individual or None if unavailable.
        """
        if not self.best_individual:
            return None

        activities = [a for a in self.best_individual['activities'] if a in self.filtered_events]

        C = set((a, b) for (a, b) in self.best_individual['C']
                if a in self.filtered_events and b in self.filtered_events)

        I = {}
        O = {}

        for a in activities:
            raw_I = self.best_individual['I'].get(a, [])
            I[a] = [inp for inp in raw_I if all(event in self.filtered_events for event in inp)]
            if not I[a]:
                I[a] = [set()]

            raw_O = self.best_individual['O'].get(a, [])
            O[a] = [out for out in raw_O if all(event in self.filtered_events for event in out)]
            if not O[a]:
                O[a] = [set()]

        return {
            'activities': activities,
            'C': C,
            'I': I,
            'O': O
        }

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

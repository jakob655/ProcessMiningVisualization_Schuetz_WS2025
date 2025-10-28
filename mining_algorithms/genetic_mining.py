import random
import threading
import uuid
import itertools

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

        petri_net = self._build_petri_net(individual)
        self.logger.debug(f"Petri net structure: {petri_net}")
        individual['_petri_net'] = petri_net
        self.petri_net = petri_net

        created_edges = set()

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

        for transition_id, data in petri_net['transitions'].items():
            if data['visible']:
                continue
            if not self.graph.contains_node(transition_id):
                self.graph.add_silent_transition(transition_id)

        for place_id in petri_net['places']:
            if not self.graph.contains_node(place_id):
                self.graph.add_place(place_id)

        for source, target in petri_net['arcs']:
            self._safe_create_edge(source, target, created_edges)

        self.logger.debug("[Graph] Finished.")

    def _build_petri_net(self, individual):
        """
        Build an internal Petri net (with silent transitions) from the given individual.
        """
        net = {
            'places': set(),                 # all unique places in the net
            'transitions': {},               # mapping transition_id -> {'inputs', 'outputs', 'visible'}
            'arcs': set(),                   # all (source, target) connections
            'initial_marking': {},           # marking of tokens at start
            'final_places': set(),           # end places of the net
            'start_buffer_places': set(),    # buffer places for start transitions
            'input_subset_map': {},          # mapping of input places to activity/subset
            'empty_input_activities': set(), # activities with no input set
        }

        # Define start and end places
        start_place = "p_start"
        end_place = "p_end"

        # Register start and end places
        self._register_place(net, start_place)
        self._register_place(net, end_place)

        # Conditional initial token:
        # - if start_nodes are defined, start_place stays empty (tokens added later)
        # - otherwise, put one token on start_place
        if hasattr(self, 'start_nodes') and self.start_nodes:
            net['initial_marking'][start_place] = 0
        else:
            net['initial_marking'][start_place] = 1

        # Register end place and buffer info
        net['final_places'].add(end_place)
        net['start_buffer_places'].add(start_place)

        # Add virtual arcs from Start/End labels (for visualization)
        net['arcs'].add(("Start", start_place))
        net['arcs'].add((end_place, "End"))

        # --- Register all visible activities as transitions ---
        for act in individual['activities']:
            self._register_transition(net, act, visible=True)

        # Initialize helper structures for mapping input/output connections
        pred_to_input_place = {}   # (pred, act) -> place_id
        activity_input_places = {} # act -> set of input places

       # Get input and output sets
        inputs = individual.get('I', {})
        outputs = individual.get('O', {})

        # Create input places for each input subset
        for act in individual['activities']:
            subsets = inputs.get(act) or []

            # Case 1: No input subsets
            if not subsets:
                net['empty_input_activities'].add(act)
                self._ensure_input_place(
                    net,
                    act,
                    "Start",
                    pred_to_input_place,
                    activity_input_places,
                    subset=set(),
                )
                continue

            # Case 2: Iterate over each input subset
            for idx, subset in enumerate(subsets):
                if not subset:
                    # Empty input subset (start candidate)
                    net['empty_input_activities'].add(act)
                    self._ensure_input_place(
                        net,
                        act,
                        "Start",
                        pred_to_input_place,
                        activity_input_places,
                        suffix=str(idx),
                        subset=set(),
                    )
                    continue

                # Normal input subset: create corresponding input place
                place_id = f"pi_{act}_{idx}_{'-'.join(sorted(subset))}"
                if place_id not in net['places']:
                    self._register_place(net, place_id)
                    self._add_arc(net, place_id, act)
                    activity_input_places.setdefault(act, set()).add(place_id)

                # Store mapping for later lookups
                net['input_subset_map'][place_id] = {'activity': act, 'subset': set(subset)}

                # Remember which input place connects a predecessor and successor
                for pred in subset:
                    pred_to_input_place[(pred, act)] = place_id

        # Initialize counter for invisible transitions (τ)
        tau_counter = itertools.count()

        # Build output places and connect via silent transitions
        for act in individual['activities']:
            out_sets = outputs.get(act) or []

            # Case 1: No output sets -> connect to end place
            if not out_sets:
                place_id = f"po_{act}_sink"
                self._register_place(net, place_id)
                self._add_arc(net, act, place_id)

                tau_id = f"tau_{next(tau_counter)}"
                self._register_transition(net, tau_id, visible=False)
                self._add_arc(net, place_id, tau_id)
                self._add_arc(net, tau_id, end_place)
                continue

            # Case 2: Iterate over each output subset
            for idx, out_set in enumerate(out_sets):
                if not out_set:
                    # Empty output set -> sink transition to end
                    place_id = f"po_{act}_sink_{idx}"
                    self._register_place(net, place_id)
                    self._add_arc(net, act, place_id)

                    tau_id = f"tau_{next(tau_counter)}"
                    self._register_transition(net, tau_id, visible=False)
                    self._add_arc(net, place_id, tau_id)
                    self._add_arc(net, tau_id, end_place)
                    continue

                # Normal output subset -> connect to successor input places
                place_id = f"po_{act}_{idx}_{'-'.join(sorted(out_set))}"
                if place_id not in net['places']:
                    self._register_place(net, place_id)
                self._add_arc(net, act, place_id)

                for succ in out_set:
                    # Find the input place for successor or create it if needed
                    target_place = pred_to_input_place.get((act, succ))
                    if target_place is None:
                        target_place = self._ensure_input_place(
                            net,
                            succ,
                            act,
                            pred_to_input_place,
                            activity_input_places,
                            subset={act},
                        )

                    # Create invisible τ-transition between po_place and pi_place
                    tau_id = f"tau_{next(tau_counter)}"
                    self._register_transition(net, tau_id, visible=False)
                    self._add_arc(net, place_id, tau_id)
                    self._add_arc(net, tau_id, target_place)

        # Connect Start place to true start activities
        for act in individual['activities']:
            if act not in self.start_nodes:
                continue

            # Ensure a valid input place for the starting activity
            target_place = self._ensure_input_place(
                net,
                act,
                "Start",
                pred_to_input_place,
                activity_input_places,
            )

            # Add one token on the input place of each start activity
            net['initial_marking'][target_place] = net['initial_marking'].get(target_place, 0) + 1
            net['start_buffer_places'].add(target_place)

            # Create τ from start_place → activity input place
            tau_id = f"tau_{next(tau_counter)}"
            self._register_transition(net, tau_id, visible=False)
            self._add_arc(net, start_place, tau_id)
            self._add_arc(net, tau_id, target_place)

        # Postprocessing:
        # Map each output place to silent transitions that follow it
        output_to_silent = {}
        for trans_id, data in net['transitions'].items():
            if not data['visible']:
                for place in data['outputs']:
                    output_to_silent.setdefault(place, []).append(trans_id)
        net['output_to_silent'] = output_to_silent

        # Identify all input places that belong to visible transitions
        visible_input_places = set()
        for trans_id, data in net['transitions'].items():
            if data['visible']:
                visible_input_places.update(data['inputs'])

        # Identify silent transitions whose outputs are invisible
        forced_silent = set()

        for trans_id, data in net['transitions'].items():
            # only invisible transitions
            if not data['visible']:
                # check if no visible outputs
                outputs = data['outputs']
                has_no_visible_outputs = all(place not in visible_input_places for place in outputs)

                if has_no_visible_outputs:
                    forced_silent.add(trans_id)

        net['forced_silent'] = forced_silent

        return net

    
    def _ensure_input_place(self, net, act, pred, pred_to_input_place, activity_input_places, suffix="", subset=None):
        """
        Ensure that an input place for (pred -> act) exists and return corresponding ID.
        Creates a new one if necessary.
        """
        if not subset:
            subset = set()
        base = f"pi_{act}"
        if suffix:
            base += f"_{suffix}"
        if subset:
            base += f"_{'-'.join(sorted(subset))}"
        place_id = base

        if place_id not in net['places']:
            self._register_place(net, place_id)
            self._add_arc(net, place_id, act)
            activity_input_places.setdefault(act, set()).add(place_id)

        # Map subset for later
        net['input_subset_map'][place_id] = {'activity': act, 'subset': set(subset)}
        if pred:
            pred_to_input_place[(pred, act)] = place_id
        return place_id
    

    @staticmethod
    def _register_place(net, place_id):
        if place_id not in net['places']:
            net['places'].add(place_id)

    @staticmethod
    def _register_transition(net, transition_id, visible):
        if transition_id not in net['transitions']:
            net['transitions'][transition_id] = {
                'inputs': set(),
                'outputs': set(),
                'visible': visible,
            }
    
    @staticmethod
    def _add_arc(net, source, target):
        net['arcs'].add((source, target))
        transition = net['transitions'].get(source)
        if transition is not None:
            transition['outputs'].add(target)
        transition = net['transitions'].get(target)
        if transition is not None:
            transition['inputs'].add(source)



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

        
        petri_net = individual.get('_petri_net')
        if petri_net is None:
            petri_net = self._build_petri_net(individual)
            individual['_petri_net'] = petri_net

        transitions = petri_net['transitions']
        places = petri_net['places']

        # Initialize all places with zero tokens and copy the initial marking
        marking = {place: 0 for place in places}
        for place, tokens in (petri_net.get('initial_marking') or {}).items():
            marking[place] = tokens

        parsed_count = 0
        trace_sequence = list(trace)

        forced_silent = petri_net.get("forced_silent", set())

        # fire any τ transitions before starting
        self._fire_silent(transitions, marking, forced_silent)

        for event in trace_sequence:
            if event not in transitions:
                continue
            # fire τ transitions before visible event
            self._fire_silent(transitions, marking, forced_silent)

            if self._is_enabled(transitions, marking, event):
                self._fire(transitions, marking, event)
                parsed_count += 1

                #  fire τ transitions after visible event
                self._fire_silent(transitions, marking, forced_silent)

            else:
                break

        # fire τ transitions final time
        self._fire_silent(transitions, marking, forced_silent)

        is_completed = (parsed_count == len(trace))
        return parsed_count, is_completed
    
    def _ensure_token(self, place_id, transitions, marking, silent_to_place,
                  depth=0, visited_places=None, visited_transitions=None):
        """
        Recursively ensure that a given place has a token..

        Returns true if the place has a token at the end.
        """

        # If token on initial place -> Done
        if marking.get(place_id, 0) > 0:
            return True

        # Safety stop, to prevent too deep recurssion
        if depth > len(transitions):
            return False

        # Initialize sets to prevent loops
        visited_places = visited_places or set()
        visited_transitions = visited_transitions or set()

        # If place already visited -> return false
        if place_id in visited_places:
            return False
        visited_places.add(place_id)

        # Check all τ-Transitionen, which have access to this place
        possible_tau = silent_to_place.get(place_id, [])
        for tau_id in possible_tau:

            # If already visited -> continue
            if tau_id in visited_transitions:
                continue

            inputs = transitions[tau_id]['inputs']

            # Copy for recurssion
            branch_places = visited_places.copy()
            branch_transitions = visited_transitions | {tau_id}

            # Check if all inputs for τ can be filled
            can_fire = True
            for p_in in inputs:
                if marking.get(p_in, 0) == 0:
                    if not self._ensure_token(p_in, transitions, marking, silent_to_place, depth + 1, branch_places, branch_transitions):
                        can_fire = False
                        break

            # If all inputs are filled -> Fire
            if can_fire and self._is_enabled(transitions, marking, tau_id):
                self._fire(transitions, marking, tau_id)

                # Target has token -> return True
                if marking.get(place_id, 0) > 0:
                    return True

        # Target has no token -> return False
        return False


    def _is_enabled(self, transitions, marking, transition_id):
        """
        Check if transition can fire.
        Transition gets enabled if all of its input places have min. one token.
        """
        transition = transitions[transition_id]
        for place in transition["inputs"]:
            if marking.get(place, 0) <= 0:
                return False
        return True

    def _fire(self, transitions, marking, transition_id):
        """
        Fire transition:
        remove/add token
        """
        transition = transitions[transition_id]

        # consume from input places
        for place in transition["inputs"]:
            marking[place] = marking.get(place, 0) - 1

        # produce for output places
        for place in transition["outputs"]:
            marking[place] = marking.get(place, 0) + 1

    def _fire_silent(self, transitions, marking, forced_silent, max_cycles=999):
        """
        Fire all enabled forced (silent) transitions until none are left or a safety cap is reached.
        This prevents deadlocks caused by pending τ transitions that must fire automatically.
        """
        cycles = 0
        transition_fired = True
        while transition_fired and cycles < max_cycles: # only continues if still firing 
            transition_fired = False
            cycles += 1
            for tau_id in forced_silent:
                if self._is_enabled(transitions, marking, tau_id):
                    self._fire(transitions, marking, tau_id)
                    transition_fired = True
        # false if possible loop
        return cycles < max_cycles

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
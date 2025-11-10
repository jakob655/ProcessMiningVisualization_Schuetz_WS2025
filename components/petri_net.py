import itertools
from collections.abc import Iterable, Sequence
from logger import get_logger

class PetriNetToolkit:
    """Utility class that encapsulates Petri net creation, helper operations, and token-game."""

    def __init__(self, start_label: str = "Start", end_label: str = "End") -> None:
        self.start_label = start_label
        self.end_label = end_label
        self.logger = get_logger("PetriNetToolkit")

    def create_base_net(self) -> tuple[dict, str, str]:
        """Create a fresh Petri net dictionary including boundary places and metadata containers."""
        net = {
            'places': set(), # all unique places in the net
            'transitions': {}, # mapping transition_id -> {'inputs', 'outputs', 'visible'}
            'arcs': set(), # all (source, target) connections
            'initial_marking': {}, # marking of tokens at start
            'final_places': set(), # end places of the net
            'start_buffer_places': set(), # buffer places for start transitions
            'input_subset_map': {}, # mapping of input places to activity/subset
            'empty_input_activities': set(), # activities with no input set
            'output_to_silent': {}, # Map each output place to silent transitions that follow it
            'forced_silent': set(), # Identify silent transitions whose outputs are invisible
            'transition_labels': {}, # mapping transition_id -> display label
        }
        # Define start and end places
        start_place = "p_start"
        end_place = "p_end"
        # Register start and end places
        self.register_place(net, start_place)
        self.register_place(net, end_place)
        
        net['initial_marking'][start_place] = 1
        
        # Register end place and buffer info
        net['final_places'].add(end_place)
        net['start_buffer_places'].add(start_place)
        
        # Add virtual arcs from Start/End labels (for visualization)
        net['arcs'].add((self.start_label, start_place))
        net['arcs'].add((end_place, self.end_label))
        return net, start_place, end_place

    def register_place(self, net: dict, place_id: str) -> None:
        """Ensure that the place exists inside the Petri net."""
        net['places'].add(place_id)

    def register_transition(self, net: dict, transition_id: str, visible: bool, label: str | None = None) -> None:
        transition = net['transitions'].setdefault(
            transition_id,
            {'inputs': set(), 'outputs': set(), 'visible': visible}
        )
        transition['visible'] = visible
        if label is None and visible:
            label = transition_id
        if label is not None:
            net.setdefault('transition_labels', {})[transition_id] = label

    def add_arc(self, net: dict, source: str, target: str) -> None:
        net['arcs'].add((source, target))
        transition = net['transitions'].get(source)
        if transition is not None:
            transition['outputs'].add(target)
        transition = net['transitions'].get(target)
        if transition is not None:
            transition['inputs'].add(source)

    def ensure_input_place(
        self,
        net: dict,
        act: str,
        pred: str,
        pred_to_input_place: dict,
        activity_input_places: dict,
        suffix: str = "",
        subset: set | None = None,
    ) -> str:
        """Ensure that an input place for (pred -> act) exists and return the corresponding ID.
        Creates a new one if necessary.
        """
        subset = subset or set()
        base = f"pi_{act}"
        if suffix:
            base += f"_{suffix}"
        if subset:
            base += f"_{'-'.join(sorted(subset))}"
        place_id = base

        if place_id not in net['places']:
            self.register_place(net, place_id)
            self.add_arc(net, place_id, act)
            activity_input_places.setdefault(act, set()).add(place_id)

        # Map subset for later
        net['input_subset_map'][place_id] = {'activity': act, 'subset': set(subset)}
        if pred:
            pred_to_input_place[(pred, act)] = place_id
        return place_id

    def build_from_genetic_individual(self, individual: dict, start_nodes: set[str]) -> dict:
        """Recreate the original Genetic Mining Petri net construction using the shared helper class."""
        net, start_place, end_place = self.create_base_net()
        
        # Register all visible activities as transitions
        for act in individual['activities']:
            self.register_transition(net, act, visible=True, label=str(act))

        # Initialize helper structures for mapping input/output connections
        pred_to_input_place: dict[tuple[str, str], str] = {}
        activity_input_places: dict[str, set[str]] = {}

        # Get input and output sets
        inputs = individual.get('I', {})
        outputs = individual.get('O', {})

        # Create input places for each input subset
        for act in individual['activities']:
            subsets = inputs.get(act) or []

            # Case 1: No input subsets
            if not subsets:
                net['empty_input_activities'].add(act)
                self.ensure_input_place(
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
                    self.ensure_input_place(
                        net,
                        act,
                        "Start",
                        pred_to_input_place,
                        activity_input_places,
                        subset=set(),
                    )
                    continue
                
                # Normal input subset: create corresponding input place
                place_id = f"pi_{act}_{idx}_{'-'.join(sorted(subset))}"
                if place_id not in net['places']:
                    self.register_place(net, place_id)
                    self.add_arc(net, place_id, act)
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
                self.register_place(net, place_id)
                self.add_arc(net, act, place_id)

                tau_id = f"tau_{next(tau_counter)}"
                self.register_transition(net, tau_id, visible=False)
                self.add_arc(net, place_id, tau_id)
                self.add_arc(net, tau_id, end_place)
                continue
            
            # Case 2: Iterate over each output subset
            for idx, out_set in enumerate(out_sets):
                if not out_set:
                    # Empty output set -> sink transition to end
                    place_id = f"po_{act}_sink_{idx}"
                    self.register_place(net, place_id)
                    self.add_arc(net, act, place_id)

                    tau_id = f"tau_{next(tau_counter)}"
                    self.register_transition(net, tau_id, visible=False)
                    self.add_arc(net, place_id, tau_id)
                    self.add_arc(net, tau_id, end_place)
                    continue
                
                # Normal output subset -> connect to successor input places
                place_id = f"po_{act}_{idx}_{'-'.join(sorted(out_set))}"
                if place_id not in net['places']:
                    self.register_place(net, place_id)
                self.add_arc(net, act, place_id)

                for succ in out_set:
                    # Find the input place for successor or create it if needed
                    target_place = pred_to_input_place.get((act, succ))
                    if target_place is None:
                        target_place = self.ensure_input_place(
                            net,
                            succ,
                            act,
                            pred_to_input_place,
                            activity_input_places,
                            subset={act},
                        )

                    # Create invisible τ-transition between po_place and pi_place
                    tau_id = f"tau_{next(tau_counter)}"
                    self.register_transition(net, tau_id, visible=False)
                    self.add_arc(net, place_id, tau_id)
                    self.add_arc(net, tau_id, target_place)

        # Connect Start place to true start activities
        for act in individual['activities']:
            if act not in start_nodes:
                continue
            
            # Ensure a valid input place for the starting activity
            target_place = self.ensure_input_place(
                net,
                act,
                "Start",
                pred_to_input_place,
                activity_input_places,
            )

            net['start_buffer_places'].add(target_place)

            for place_id, meta in net['input_subset_map'].items():
                if meta.get('activity') == act:
                    subset = meta.get('subset') or set()
                    if subset == {act}:
                        net['initial_marking'][place_id] = net['initial_marking'].get(place_id, 0) + 1
                        net['start_buffer_places'].add(place_id)

            # Create τ from start_place → activity input place
            tau_id = f"tau_{next(tau_counter)}"
            self.register_transition(net, tau_id, visible=False)
            self.add_arc(net, start_place, tau_id)
            self.add_arc(net, tau_id, target_place)

        self.finalize_net(net)
        return net

    def finalize_net(self, net: dict) -> None:
        """Populate helper mappings for silent transitions and forced firing."""
        
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

    def simulate_trace(self, net: dict, trace: Sequence[str], start_events: set[str]) -> tuple[int, bool]:
        """Token-game simulation for a trace on the provided Petri net."""
        transitions = net['transitions']
        places = net['places']

        # Initialize all places with zero tokens and copy the initial marking
        marking = {place: 0 for place in places}
        for place, tokens in (net.get('initial_marking') or {}).items():
            marking[place] = tokens

        parsed_count = 0
        trace_sequence = list(trace)

        forced_silent = set(net.get('forced_silent') or set())
        silent_to_place = dict(net.get('output_to_silent') or {})
        final_places = set(net.get('final_places') or set())
        start_buffer_places = set(net.get('start_buffer_places') or set())
        input_subset_map = net.get('input_subset_map') or {}

        # Fire forced τ transitions before start
        self._fire_silent(transitions, marking, forced_silent)

         # Go through the trace step by step
        for event in trace_sequence:
            transition = transitions.get(event)
            if transition is None or not transition.get('visible', False):
                break # if not a valid visible transition -> skip

            required_inputs = list(transition['inputs'])
            event_enabled = True
            skip_current_event = False

            for place_id in required_inputs:
                # If it is a start activity -> drop a token there if needed
                subset_meta = input_subset_map.get(place_id) or {}
                subset = subset_meta.get('subset') or set()
                is_start_input = (
                    subset_meta.get('activity') == event
                    and not subset
                    and event in start_events
                )

                # Make sure input place actually has a token
                if marking.get(place_id, 0) == 0:
                    if not self._ensure_token(place_id, transitions, marking, silent_to_place):
                        if is_start_input:
                            skip_current_event = True
                            break
                        event_enabled = False
                        break

            if skip_current_event:
                continue

            if not event_enabled or not self._is_enabled(transitions, marking, event):
                break # cant fire this event, stop here
            
            # Fire transition
            self._fire(transitions, marking, event)
            parsed_count += 1

            # Fire forced τ transitions 
            self._fire_silent(transitions, marking, forced_silent)

        #  Last check for silent transitions
        self._fire_silent(transitions, marking, forced_silent)

        # Keep only places that still have tokens
        remaining_tokens = {place: tokens for place, tokens in marking.items() if tokens > 0}

        # Count tokens that are NOT in final or start places
        non_final_tokens = 0
        for place, tokens in remaining_tokens.items():
            if place not in final_places and place not in start_buffer_places:
                non_final_tokens += tokens

        # Count tokens that are in final places
        final_tokens = 0
        for place, tokens in remaining_tokens.items():
            if place in final_places:
                final_tokens += tokens

        # Trace is complete if all events were fired and no tokens left
        final_ok = True if not final_places else (final_tokens == 1 or not trace_sequence)
        is_completed = (parsed_count == len(trace_sequence) and non_final_tokens == 0 and final_ok)

        return parsed_count, is_completed

    def _ensure_token(
        self,
        place_id: str,
        transitions: dict,
        marking: dict,
        silent_to_place: dict,
        depth: int = 0,
        visited_places: set | None = None,
        visited_transitions: set | None = None,
    ) -> bool:
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

    @staticmethod
    def _is_enabled(transitions: dict, marking: dict, transition_id: str) -> bool:
        """
        Check if transition can fire.
        Transition gets enabled if all of its input places have min. one token.
        """
        transition = transitions[transition_id]
        for place in transition['inputs']:
            if marking.get(place, 0) <= 0:
                return False
        return True

    @staticmethod
    def _fire(transitions: dict, marking: dict, transition_id: str) -> None:
        """
        Fire transition:
        remove/add token
        """
        transition = transitions[transition_id]
        
        # consume from input places
        for place in transition['inputs']:
            marking[place] = marking.get(place, 0) - 1
            
        # produce for output places
        for place in transition['outputs']:
            marking[place] = marking.get(place, 0) + 1

    def _fire_silent(self, transitions: dict, marking: dict, forced_silent: set[str], max_cycles: int = 999) -> bool:
        """
        Fire all enabled forced silent transitions until none are left or a safety cap is reached.
        This prevents deadlocks
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


def add_petri_net_to_graph(
    graph,
    petri_net: dict,
    visible_labels: Iterable[str],
    node_stats_map: dict[str, dict],
    frequency_map: dict[str, int],
    logger=None,
) -> None:
    """Populate a visualization graph with the contents of a Petri net."""
    logger = logger or get_logger("PetriNetToolkit")
    created_edges = set()
    labels = petri_net.get('transition_labels', {})
    allowed_labels = set(visible_labels) if visible_labels is not None else None

    for transition_id, data in petri_net['transitions'].items():
        label = labels.get(transition_id, transition_id)
        if not data.get('visible', False):
            if not graph.contains_node(transition_id):
                graph.add_silent_transition(transition_id)
            continue

        if allowed_labels is not None and label not in allowed_labels:
            continue

        stats = node_stats_map.get(label) or node_stats_map.get(transition_id) or {}
        abs_freq = frequency_map.get(label, frequency_map.get(transition_id, 0))
        graph.add_event(
            transition_id,
            spm=stats.get("spm", 0.0),
            normalized_frequency=stats.get("frequency", 0.0),
            absolute_frequency=abs_freq,
        )

    for transition_id, data in petri_net['transitions'].items():
        if data.get('visible', False):
            continue
        if not graph.contains_node(transition_id):
            graph.add_silent_transition(transition_id)

    for place_id in petri_net['places']:
        if not graph.contains_node(place_id):
            graph.add_place(place_id)

    for source, target in petri_net['arcs']:
        _safe_create_edge(graph, source, target, created_edges, logger)


def _safe_create_edge(graph, source: str, target: str, created_edges: set[tuple[str, str]], logger) -> None:
    if graph.contains_node(source) and graph.contains_node(target):
        edge = (source, target)
        if edge not in created_edges:
            graph.create_edge(source, target)
            created_edges.add(edge)
    else:
        if logger:
            logger.debug(f"[PetriNetToolkit] Skipped edge {source} -> {target} (missing node)")


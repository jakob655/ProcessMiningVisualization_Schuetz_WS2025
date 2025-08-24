import streamlit as st

from components.number_input_slider import number_input_slider
from ui.base_algorithm_ui.base_algorithm_view import BaseAlgorithmView


class GeneticMinerView(BaseAlgorithmView):
    """View for the Genetic Miner algorithm."""

    def render_log_filter_extensions(self, sidebar_values: dict[str, any]) -> None:
        st.write("### **Genetic Mining**")

        with st.expander("⚙️ **Settings:**", expanded=False):
            number_input_slider(
                label="Population Size",
                min_value=sidebar_values["population_size"][0],
                max_value=sidebar_values["population_size"][1],
                key="population_size",
                help="Number of candidate models (individuals) in each generation. Larger populations improve diversity and search coverage, but increase runtime cost.",
            )

            number_input_slider(
                label="Max Generations",
                min_value=sidebar_values["max_generations"][0],
                max_value=sidebar_values["max_generations"][1],
                key="max_generations",
                help="Maximum number of generations to run. The run may terminate earlier if optimal fitness is reached or the fitness stagnates.",
            )

            number_input_slider(
                label="Crossover Rate",
                min_value=sidebar_values["crossover_rate"][0],
                max_value=sidebar_values["crossover_rate"][1],
                key="crossover_rate",
                help="Probability of recombining two parents. Typical values: 0.6-0.9. Low values reduce exploration, excessively high values may reduce diversity.",
            )

            number_input_slider(
                label="Mutation Rate",
                min_value=sidebar_values["mutation_rate"][0],
                max_value=sidebar_values["mutation_rate"][1],
                key="mutation_rate",
                help="Probability of introducing random variation per activity. Typical values: 0.01-0.2. Low values reduce diversity, high values disrupt convergence.",
            )

            number_input_slider(
                label="Elitism Rate",
                min_value=sidebar_values["elitism_rate"][0],
                max_value=sidebar_values["elitism_rate"][1],
                key="elitism_rate",
                help="Fraction of top individuals copied unchanged into next generation. Typical values: 0.01-0.2. Higher values (>0.3) risk stagnation.",
            )

            number_input_slider(
                label="Tournament Size",
                min_value=sidebar_values["tournament_size"][0],
                max_value=sidebar_values["tournament_size"][1],
                key="tournament_size",
                help="Number of candidates competing in each tournament selection. Larger sizes increase selection pressure (favoring fitter individuals), smaller sizes maintain diversity.",
            )

            st.slider(
                label="Power Value",
                min_value=sidebar_values["power_value"][0],
                max_value=sidebar_values["power_value"][1],
                step=2,
                key="power_value",
                help="Odd number controlling sparsity of initial causal matrices. Dependency measures are raised to this power, so higher values create initial populations with less causal relations."
            )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Run Genetic Mining", type="secondary"):
                st.session_state.rerun_genetic_miner = True

        with col2:
            if st.button("Cancel active Run", type="primary"):
                st.session_state.cancel_run = True
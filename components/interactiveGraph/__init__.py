import os

import streamlit as st
import streamlit.components.v1 as components

from graphs.visualization.base_graph import BaseGraph

# Template for the component from https://docs.streamlit.io/library/components/publish and https://github.com/streamlit/component-template/tree/master/template/my_component

_RELEASE = True
_COMPONENT_NAME = "interactive-graph"

if not _RELEASE:
    _component_func = components.declare_component(
        _COMPONENT_NAME,
        url="http://localhost:3000",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(_COMPONENT_NAME, path=build_dir)


def interactiveGraph(
        graph: BaseGraph, onNodeClick, onEdgeClick, key="interactiveGraph", height=600
) -> None:
    state_name = f"previous_clickId-{key}"
    edge_state_name = f"previous_edgeClickId-{key}"

    if state_name not in st.session_state:
        st.session_state[state_name] = ""
    if edge_state_name not in st.session_state:
        st.session_state[edge_state_name] = ""

    component_value = _component_func(
        graphviz_string=graph.get_graphviz_string(), key=key, height=height
    )

    del st.session_state[key]

    if (
        component_value is not None
        and component_value.get("clickId", "") != ""
        and component_value["clickId"] != st.session_state[state_name]
    ):
        st.session_state[state_name] = component_value["clickId"]
        node_name, description = graph.node_to_string(component_value["nodeId"])
        onNodeClick(node_name, description)

    if (
        component_value is not None
        and component_value.get("edgeClickId", "") != ""
        and component_value["edgeClickId"] != st.session_state[edge_state_name]
    ):
        st.session_state[edge_state_name] = component_value["edgeClickId"]
        source = component_value.get("source", "")
        target = component_value.get("target", "")

        edge_description = graph.edge_to_string(source, target)
        onEdgeClick(source, target, edge_description)

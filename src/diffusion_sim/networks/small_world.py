"""
Small-world network generation using Watts-Strogatz model.
"""

import networkx as nx

from ..core.exceptions import NetworkBuildError
from ..core.schemas import AgentState, NetworkConfig


class SmallWorldNetworkBuilder:
    """Builds a Watts-Strogatz small-world network."""

    def build(self, config: NetworkConfig) -> nx.Graph:
        """
        Build a small-world network using the Watts-Strogatz model.

        The Watts-Strogatz model creates a network with high clustering
        (like regular lattices) and short path lengths (like random graphs).

        Args:
            config: Network configuration with n, k, p parameters

        Returns:
            NetworkX graph with initialized agent states

        Raises:
            NetworkBuildError: If network generation fails
        """
        if config.network_type != "small_world":
            raise NetworkBuildError(
                f"SmallWorldNetworkBuilder requires network_type='small_world', "
                f"got '{config.network_type}'"
            )

        # Validate parameters
        if config.k >= config.n:
            raise NetworkBuildError(
                f"Average degree k={config.k} must be less than n={config.n}"
            )

        if config.k % 2 != 0:
            raise NetworkBuildError(
                f"Average degree k={config.k} must be even for Watts-Strogatz model"
            )

        try:
            # Generate Watts-Strogatz small-world graph
            graph = nx.watts_strogatz_graph(
                n=config.n, k=config.k, p=config.p, seed=config.seed
            )

            # Initialize agent states for all nodes
            for node_id in graph.nodes():
                agent_state = AgentState(agent_id=node_id)
                graph.nodes[node_id]["agent_state"] = agent_state

            return graph

        except Exception as e:
            raise NetworkBuildError(f"Failed to generate small-world network: {e}")

    def get_network_stats(self, graph: nx.Graph) -> dict:
        """
        Compute basic network statistics.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary with network statistics
        """
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
            "clustering_coefficient": nx.average_clustering(graph),
            "avg_shortest_path": (
                nx.average_shortest_path_length(graph)
                if nx.is_connected(graph)
                else "Graph not connected"
            ),
        }
